package com.arcrobotics.ftclib.vision;

import com.arcrobotics.ftclib.vision.UGContourRingPipeline.Height;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static com.arcrobotics.ftclib.vision.TestCase.TestCaseRings;
import static com.arcrobotics.ftclib.vision.UGAdvancedHighGoalPipeline.Target;
import static com.arcrobotics.ftclib.vision.VisionTestHelper.loadMatFromBGR;
import static com.arcrobotics.ftclib.vision.VisionTestHelper.saveMatAsRGB;
import static com.google.common.truth.Truth.assertThat;

public class VisionTutorial {

    // Load x64 OpenCV Library dll
    static {
        try {
            System.load("C:/opencv/build/java/x64/opencv_java412.dll");
            // https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.3/
            // https://sourceforge.net/projects/opencvlibrary/files/4.1.2/opencv-4.1.2-vc14_vc15.exe/
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load.\n" + e);
            System.err.println("For windows 10, download OpenCV Library from:");
            //System.err.println("https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.3/");
            System.err.println("https://sourceforge.net/projects/opencvlibrary/files/4.1.2/opencv-4.1.2-vc14_vc15.exe/");
            // Just the 50 MB dll from google docs
			System.err.println("https://drive.google.com/file/d/1vgO4UNozY0Zq2aSPP1nH90zFbTrPlg1s/view?usp=sharing");			
            System.err.println("https://opencv.org/releases/");
            System.err.println("And extract to your C:\\ drive");

            System.exit(1);
        }
    }


    String IMAGE_READ_PATH = "./TestData/openCV_input/";
    String IMAGE_WRITE_PATH = "./TestData/openCV_output/";
//    Mat inputMat = new Mat();
    ArrayList<String> faceImages = new ArrayList<>();

    @Before
    public void initialize() {
        faceImages.add("Tom_Cruise_0001.jpg");
        faceImages.add("Holly_Hunter_0004.jpg");
        faceImages.add("Holly_Hunter_0001.jpg");
        faceImages.add("Al_Sharpton_0002.jpg");
        faceImages.add("Keanu_Reeves_0002.jpg ");
    }


    // https://docs.opencv.org/master/d9/d52/tutorial_java_dev_intro.html
    @Test
    public void faceDetection() {
        String TEST_TYPE = "FaceDetection";
        String IMAGE_PATH_SUBFOLDER = "faceDetection/";
        String READ_PATH = IMAGE_READ_PATH + IMAGE_PATH_SUBFOLDER;
        String WRITE_PATH = IMAGE_WRITE_PATH + IMAGE_PATH_SUBFOLDER;
        Mat image = new Mat();

        for (String imageName : faceImages) {
            image = loadMatFromBGR(READ_PATH + imageName);

            CascadeClassifier faceDetector = new CascadeClassifier("./TestData/lbpcascade_frontalface.xml");
            MatOfRect faceDetections = new MatOfRect();
            faceDetector.detectMultiScale(image, faceDetections);

            System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));

            // Draw a bounding box around each face
            for (Rect rect : faceDetections.toArray()) {
                Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width,
                        rect.y + rect.height), new Scalar(0, 255, 0));
            }

            saveMatAsRGB(WRITE_PATH + TEST_TYPE + "_" + imageName, image);
        }
    }


    @Ignore("Test for reference only")
    @Test
    public void colorThresholding() {
        String TEST_TYPE = "ColorThresholding";
        String IMAGE_PATH_SUBFOLDER = "ug1/";
        String READ_PATH = IMAGE_READ_PATH + IMAGE_PATH_SUBFOLDER;
        Mat inputMat = new Mat();
        inputMat = loadMatFromBGR(READ_PATH + "blue_ring_4.jpg" );

        Mat yCbCrChan2Mat = new Mat();
        Mat thresholdMat = new Mat();
        Mat all = new Mat();
        List<MatOfPoint> contoursList = new ArrayList<>();

        Imgproc.cvtColor(inputMat, yCbCrChan2Mat, Imgproc.COLOR_RGB2YCrCb);//converts rgb to ycrcb
        Core.extractChannel(yCbCrChan2Mat, yCbCrChan2Mat, 2);//takes cb difference and stores

        //b&w
        Imgproc.threshold(yCbCrChan2Mat, thresholdMat, 100, 255, Imgproc.THRESH_BINARY_INV);

        //outline/contour
        Imgproc.findContours(thresholdMat, contoursList, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        yCbCrChan2Mat.copyTo(all);//copies mat object
        Imgproc.drawContours(all, contoursList, -1, new Scalar(255, 0, 0), 3, 8);//draws blue contours

        Imgproc.rectangle(
                all,
                new Point(
                        inputMat.cols()*0.25,
                        inputMat.rows()*0.25),
                new Point(
                        inputMat.cols()*0.75,
                        inputMat.rows()*0.75),
                new Scalar(0, 255, 0), 3);


        Imgcodecs.imwrite(IMAGE_WRITE_PATH + TEST_TYPE + "_" + "yCbCr.jpg", yCbCrChan2Mat);
        Imgcodecs.imwrite(IMAGE_WRITE_PATH + TEST_TYPE + "_" + "threshold.jpg", thresholdMat);
        Imgcodecs.imwrite(IMAGE_WRITE_PATH + TEST_TYPE + "_" + "all.jpg", all);
    }








}


