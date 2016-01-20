#include <CameraIntrinsicsLoader.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include "AsusXtionImageProducer.h"

using namespace cv;
using namespace std;

const char *usage =
        " \nexample command line for calibration from a live feed.\n"
                "   calibration  -w=4 -h=5 -s=0.025 -o=camera.yml 1 2\n";


const char *liveCaptureHelp =
        "When the live video from camera is used as input, the following hot-keys may be used:\n"
                "  <ESC>, 'q' - quit the program\n"
                "  'g' - start capturing images\n"
                "  'u' - switch undistortion on/off\n";

static void help() {
    printf("This is a stereo calibration program for the Asus Xtion Pro Live camera.\n"
                   "Usage: calibration\n"
                   "     -w=<board_width>         # the number of inner corners per one of board dimension\n"
                   "     -h=<board_height>        # the number of inner corners per another board dimension\n"
                   "     [-ir]                    # Use the IR channels of the Asus Xtion camera (default is RGB)\n"
                   "     [-pt=<pattern>]          # the type of pattern: chessboard or circles' grid\n"
                   "     [-n=<number_of_frames>]  # the number of frames to use for calibration (default 15) \n"
                   "     [-d=<delay>]             # a minimum delay in ms between subsequent attempts to capture a next view\n"
                   "     [-s=<squareSize>]        # square size in some user-defined units (1 by default)\n"
                   "     [-o=<out_camera_params>] # the output filename for extrinsic parameters\n"
                   "     [-zt]                    # assume zero tangential distortion\n"
                   "     [-a=<aspectRatio>]       # fix aspect ratio (fx/fy)\n"
                   "     [-p]                     # fix the principal point at the center\n"
                   "     [-su]                    # show undistorted images after calibration\n"
                   "     [input_data]             # input data, a list of the camera indices (use pcl_openni2_viewer -l)\n"
                   "\n");
    printf("\n%s", usage);
    printf("\n%s", liveCaptureHelp);
}

enum {
    DETECTION = 0, CAPTURING = 1, CALIBRATED = 2
};

enum Pattern {
    CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID
};

static double computeReprojectionErrors(
        const vector<vector<Point3f> > &objectPoints,
        const vector<vector<Point2f> > &imagePoints,
        const vector<Mat> &rvecs, const vector<Mat> &tvecs,
        const Mat &cameraMatrix, const Mat &distCoeffs,
        vector<float> &perViewErrors) {
    vector<Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for (i = 0; i < (int) objectPoints.size(); i++) {
        projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i],
                      cameraMatrix, distCoeffs, imagePoints2);
        err = norm(Mat(imagePoints[i]), Mat(imagePoints2), NORM_L2);
        int n = (int) objectPoints[i].size();
        perViewErrors[i] = (float) std::sqrt(err * err / n);
        totalErr += err * err;
        totalPoints += n;
    }

    return std::sqrt(totalErr / totalPoints);
}

static void calcChessboardCorners(Size boardSize, float squareSize, vector<Point3f> &corners,
                                  Pattern patternType = CHESSBOARD) {
    corners.resize(0);

    switch (patternType) {
        case CHESSBOARD:
        case CIRCLES_GRID:
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    corners.push_back(Point3f(float(j * squareSize),
                                              float(i * squareSize), 0));
            break;

        case ASYMMETRIC_CIRCLES_GRID:
            for (int i = 0; i < boardSize.height; i++)
                for (int j = 0; j < boardSize.width; j++)
                    corners.push_back(Point3f(float((2 * j + i % 2) * squareSize),
                                              float(i * squareSize), 0));
            break;

        default:
            CV_Error(Error::StsBadArg, "Unknown pattern type\n");
    }
}

static bool runCalibration(vector<vector<Point2f> > image_points_0,
                           vector<vector<Point2f> > image_points_1,
                           Size imageSize, Size boardSize, Pattern patternType,
                           float squareSize, float aspectRatio,
                           int flags, Mat &camera_matrix_0, Mat &camera_matrix_1,
                           Mat &dist_coeffs_0, Mat &dist_coeffs_1,
                           vector<Mat> &rvecs, vector<Mat> &tvecs,
                           vector<float> &reprojErrs,
                           double &totalAvgErr) {
    // TODO: Load calibrations from disk

    camera_matrix_0 = Mat::eye(3, 3, CV_64F);
    camera_matrix_1 = Mat::eye(3, 3, CV_64F);
    if (flags & CALIB_FIX_ASPECT_RATIO) {
        camera_matrix_0.at<double>(0, 0) = aspectRatio;
        camera_matrix_1.at<double>(0, 0) = aspectRatio;
    }

    dist_coeffs_0 = Mat::zeros(8, 1, CV_64F);
    dist_coeffs_1 = Mat::zeros(8, 1, CV_64F);

    vector<vector<Point3f> > objectPoints(1);
    calcChessboardCorners(boardSize, squareSize, objectPoints[0], patternType);

    objectPoints.resize(image_points_0.size(), objectPoints[0]);

    Mat R, T, E, F;

    double rms =stereoCalibrate(objectPoints,
                                image_points_0,
                                image_points_1,
                                camera_matrix_0,
                                dist_coeffs_0,
                                camera_matrix_1,
                                dist_coeffs_1,
                                imageSize,
                                R,
                                T,
                                E,
                                F,
                                flags | CALIB_FIX_INTRINSIC);

    printf("RMS error reported by stereo calibrate: %g\n", rms);

    bool ok = checkRange(R) && checkRange(T);

    totalAvgErr = computeReprojectionErrors(objectPoints, image_points_0,
                                            rvecs, tvecs, camera_matrix_0, dist_coeffs_0, reprojErrs);

    return ok;
}


static void saveCameraParams(const string &filename,
                             Size imageSize, Size boardSize,
                             float squareSize, float aspectRatio, int flags,
                             const Mat &camera_matrix_0, const Mat & camera_matrix_1,
                             const Mat &dist_coeffs_0, const Mat & dist_coeffs_1,
                             const vector<Mat> &rvecs, const vector<Mat> &tvecs,
                             const vector<float> &reprojErrs,
                             double totalAvgErr) {
    FileStorage fs(filename, FileStorage::WRITE);

    time_t tt;
    time(&tt);
    struct tm *t2 = localtime(&tt);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    if (!rvecs.empty() || !reprojErrs.empty())
        fs << "nframes" << (int) std::max(rvecs.size(), reprojErrs.size());
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << boardSize.width;
    fs << "board_height" << boardSize.height;
    fs << "square_size" << squareSize;

    if (flags & CALIB_FIX_ASPECT_RATIO)
        fs << "aspectRatio" << aspectRatio;

    if (flags != 0) {
        sprintf(buf, "flags: %s%s%s%s",
                flags & CALIB_USE_INTRINSIC_GUESS ? "+use_intrinsic_guess" : "",
                flags & CALIB_FIX_ASPECT_RATIO ? "+fix_aspectRatio" : "",
                flags & CALIB_FIX_PRINCIPAL_POINT ? "+fix_principal_point" : "",
                flags & CALIB_ZERO_TANGENT_DIST ? "+zero_tangent_dist" : "");
        //cvWriteComment( *fs, buf, 0 );
    }

    fs << "flags" << flags;

    fs << "camera_matrix_1" << camera_matrix_0;
    fs << "camera_matrix_2" << camera_matrix_1;
    fs << "distortion_coefficients_1" << dist_coeffs_0;
    fs << "distortion_coefficients_2" << dist_coeffs_1;

    fs << "avg_reprojection_error" << totalAvgErr;
    if (!reprojErrs.empty())
        fs << "per_view_reprojection_errors" << Mat(reprojErrs);

    if (!rvecs.empty() && !tvecs.empty()) {
        CV_Assert(rvecs[0].type() == tvecs[0].type());
        Mat bigmat((int) rvecs.size(), 6, rvecs[0].type());
        for (int i = 0; i < (int) rvecs.size(); i++) {
            Mat r = bigmat(Range(i, i + 1), Range(0, 3));
            Mat t = bigmat(Range(i, i + 1), Range(3, 6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            //*.t() is MatExpr (not Mat) so we can use assignment operator
            r = rvecs[i].t();
            t = tvecs[i].t();
        }
        //cvWriteComment( *fs, "a set of 6-tuples (rotation vector + translation vector) for each view", 0 );
        fs << "extrinsic_parameters" << bigmat;
    }
}


static bool runAndSave(const string &outputFilename,
                       const vector<vector<Point2f> > &image_points_0,
                       const vector<vector<Point2f> > &image_points_1,
                       Size imageSize, Size boardSize, Pattern patternType, float squareSize,
                       float aspectRatio, int flags, Mat &camera_matrix_0, Mat &camera_matrix_1,
                       Mat &dist_coeffs_0, Mat &dist_coeffs_1) {
    vector<Mat> rvecs, tvecs;
    vector<float> reprojErrs;
    double totalAvgErr = 0;

    bool ok = runCalibration(image_points_0, image_points_1, imageSize, boardSize, patternType, squareSize,
                             aspectRatio, flags,
                             camera_matrix_0, camera_matrix_1, dist_coeffs_0, dist_coeffs_1,
                             rvecs, tvecs, reprojErrs, totalAvgErr);
    printf("%s. avg reprojection error = %.2f\n",
           ok ? "Calibration succeeded" : "Calibration failed",
           totalAvgErr);

    if (ok)
        saveCameraParams(outputFilename, imageSize,
                         boardSize, squareSize, aspectRatio,
                         flags, camera_matrix_0, camera_matrix_1, dist_coeffs_0, dist_coeffs_1,
                         rvecs,
                         tvecs,
                         reprojErrs,
                         totalAvgErr);
    return ok;
}


int main(int argc, const char *const *argv) {
    Size boardSize, imageSize;
    float squareSize, aspectRatio;
    Mat camera_matrix_0, camera_matrix_1, dist_coeffs_0, dist_coeffs_1;
    string outputFilename;

    int i, nframes;
    bool undistortImage = false;
    bool ir = false;
    int flags = 0;
    bool showUndistorted;
    int delay;
    clock_t prevTimestamp = 0;
    int mode = DETECTION;
    int cameraId_0 = -1;
    int cameraId_1 = -1;
    vector<vector<Point2f> > image_points_0;
    vector<vector<Point2f> > image_points_1;
    Pattern pattern = CHESSBOARD;

    cv::CommandLineParser parser(argc, argv,
                                 "{help ||}{w||}{h||}{pt|chessboard|}{n|15|}{d|3000|}{s|1|}{o|out_camera_data.yml|}"
                                         "{ir||}{zt||}{a|1|}{p||}{su||}"
                                         "{@input_data_1|-1|}{@input_data_2|-1|}");
    if (parser.has("help")) {
        help();
        return 0;
    }
    boardSize.width = parser.get<int>("w");
    boardSize.height = parser.get<int>("h");
    string val = parser.get<string>("pt");
    if (val == "circles")
        pattern = CIRCLES_GRID;
    else if (val == "acircles")
        pattern = ASYMMETRIC_CIRCLES_GRID;
    else if (val == "chessboard")
        pattern = CHESSBOARD;
    else
        return fprintf(stderr, "Invalid pattern type: must be chessboard or circles\n"), -1;
    ir = parser.has("ir");
    squareSize = parser.get<float>("s");
    nframes = parser.get<int>("n");
    aspectRatio = parser.get<float>("a");
    delay = parser.get<int>("d");
    if (parser.has("a"))
        flags |= CALIB_FIX_ASPECT_RATIO;
    if (parser.has("zt"))
        flags |= CALIB_ZERO_TANGENT_DIST;
    if (parser.has("p"))
        flags |= CALIB_FIX_PRINCIPAL_POINT;

    flags |= CALIB_FIX_K1;
    flags |= CALIB_FIX_K2;
    flags |= CALIB_FIX_K3;
    flags |= CALIB_FIX_K4;
    flags |= CALIB_FIX_K5;
    flags |= CALIB_FIX_K6;
    outputFilename = parser.get<string>("o");
    showUndistorted = parser.has("su");
    cameraId_0 = parser.get<int>("@input_data_1");
    cameraId_1 = parser.get<int>("@input_data_2");
//    auto input_data = parser.get<string>("@input_data_1");
//    if (isdigit(input_data[0]))
//        cameraId_0 = static_cast<int>(input_data[0]);
//
//    auto input_data_2 = parser.get<string>("@input_data_2");
//
//    if(isdigit(input_data_2[0]))
//        cameraId_1 = static_cast<int>(input_data_2[0]);

    if (!parser.check()) {
        help();
        parser.printErrors();
        return -1;
    }
    if (squareSize <= 0)
        return fprintf(stderr, "Invalid board square width\n"), -1;
    if (nframes <= 3)
        return printf("Invalid number of images\n"), -1;
    if (aspectRatio <= 0)
        return printf("Invalid aspect ratio\n"), -1;
    if (delay <= 0)
        return printf("Invalid delay\n"), -1;
    if (boardSize.width <= 0)
        return fprintf(stderr, "Invalid board width\n"), -1;
    if (boardSize.height <= 0)
        return fprintf(stderr, "Invalid board height\n"), -1;

    // TODO: Open Asus Xtion camera
    // return an error if no camera found
    // capture.open(cameraId);
    AsusXtionImageProducer capture_0(ir);
    AsusXtionImageProducer capture_1(ir);
    capture_0.open(cameraId_0);
    capture_1.open(cameraId_1);

    if (capture_0.isOpened() && capture_1.isOpened()) {
        printf("%s", liveCaptureHelp);
        // Load in the camera and dist matrices
        bool loaded = false;
        loaded = CameraIntrinsicsLoader::getIntrinsics(cameraId_0, camera_matrix_0, dist_coeffs_0);
        loaded = loaded && CameraIntrinsicsLoader::getIntrinsics(cameraId_1, camera_matrix_1, dist_coeffs_1);

        if(!loaded)
            printf("Failed to load intrinsic camera parameters\n");
    }

    namedWindow("Image 1 View", 1);
    namedWindow("Image 2 View", 1);

    for (i = 0; ; i++) {
        Mat view_0, view_1, view_gray_0, view_gray_1;
        bool blink = false;

        // TODO; Get image from Asus Xtion and convert to OpenCV format
        if (capture_0.isOpened()) {
            Mat temp_view_0 = capture_0.getImage();
            Mat temp_view_1 = capture_1.getImage();
            temp_view_0.copyTo(view_0);
            temp_view_1.copyTo(view_1);
        }

        imageSize = view_0.size();

        vector<Point2f> point_buff_0;
        vector<Point2f> point_buff_1;

        if(!ir) {
            cvtColor(view_0, view_gray_0, COLOR_BGR2GRAY);
            cvtColor(view_1, view_gray_1, COLOR_BGR2GRAY);
        }

        bool found = false;
        switch (pattern) {
            case CHESSBOARD:
                found = findChessboardCorners(view_gray_0, boardSize, point_buff_0,
                                              CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK |
                                              CALIB_CB_NORMALIZE_IMAGE);

                found = found && findChessboardCorners(view_gray_1, boardSize, point_buff_1,
                                              CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_FAST_CHECK |
                                              CALIB_CB_NORMALIZE_IMAGE);
                break;
            case CIRCLES_GRID:
                found = findCirclesGrid(view_0, boardSize, point_buff_0);
                found = found && findCirclesGrid(view_1, boardSize, point_buff_1);
                break;
            case ASYMMETRIC_CIRCLES_GRID:
                found = findCirclesGrid(view_0, boardSize, point_buff_0, CALIB_CB_ASYMMETRIC_GRID);
                found = found && findCirclesGrid(view_1, boardSize, point_buff_1, CALIB_CB_ASYMMETRIC_GRID);
                break;
            default:
                return fprintf(stderr, "Unknown pattern type\n"), -1;
        }

        // improve the found corners' coordinate accuracy
        if (pattern == CHESSBOARD && found)
            if(ir) {
                cornerSubPix(view_0, point_buff_0, Size(11, 11),
                             Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
                cornerSubPix(view_1, point_buff_1, Size(11, 11),
                             Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            }
            else {
                cornerSubPix(view_gray_0, point_buff_0, Size(11, 11),
                             Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
                cornerSubPix(view_gray_1, point_buff_1, Size(11, 11),
                             Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
            }

        if (mode == CAPTURING && found && (clock() - prevTimestamp > delay * 1e-3 * CLOCKS_PER_SEC) ) {
            image_points_0.push_back(point_buff_0);
            prevTimestamp = clock();
            blink = capture_0.isOpened() && capture_1.isOpened();
        }

        if (found) {
            drawChessboardCorners(view_0, boardSize, Mat(point_buff_0), found);
            drawChessboardCorners(view_1, boardSize, Mat(point_buff_1), found);
        }

        string msg = mode == CAPTURING ? "100/100" :
                     mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
        int baseLine = 0;
        Size textSize = getTextSize(msg, 1, 1, 1, &baseLine);
        Point textOrigin_0(view_0.cols - 2 * textSize.width - 10, view_0.rows - 2 * baseLine - 10);
        Point textOrigin_1(view_1.cols - 2 * textSize.width - 10, view_1.rows - 2 * baseLine - 10);

        if (mode == CAPTURING) {
            if (undistortImage)
                msg = format("%d/%d Undist", (int) image_points_0.size(), nframes);
            else
                msg = format("%d/%d", (int) image_points_0.size(), nframes);
        }

        putText(view_0, msg, textOrigin_0, 1, 1,
                mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

        putText(view_1, msg, textOrigin_1, 1, 1,
                mode != CALIBRATED ? Scalar(0, 0, 255) : Scalar(0, 255, 0));

        // Blink
        if(blink) {
            bitwise_not(view_0, view_0);
            bitwise_not(view_1, view_1);
        }

        if (mode == CALIBRATED && undistortImage) {
            Mat temp = view_0.clone();
            undistort(temp, view_0, camera_matrix_0, dist_coeffs_0);
            temp = view_1.clone();
            undistort(temp, view_1, camera_matrix_1, dist_coeffs_1);
        }

        imshow("Image 1 View", view_0);
        imshow("Image 2 View", view_1);

        int key = 0xff & waitKey(50);

        if ((key & 255) == 27)
            break;

        if (key == 'u' && mode == CALIBRATED)
            undistortImage = !undistortImage;

        if (key == 'g') {
            mode = CAPTURING;
            image_points_0.clear();
            image_points_1.clear();
        }

        if (mode == CAPTURING && image_points_0.size() >= (unsigned) nframes &&
                image_points_1.size() >= (unsigned) nframes) {
            if (runAndSave(outputFilename, image_points_0, image_points_1, imageSize,
                           boardSize, pattern, squareSize, aspectRatio,
                           flags, camera_matrix_0, camera_matrix_1, dist_coeffs_0, dist_coeffs_1))
                mode = CALIBRATED;
            else
                mode = DETECTION;
        }
    }

    return 0;
}