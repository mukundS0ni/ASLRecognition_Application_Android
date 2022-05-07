package com.example.imagepro;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import org.checkerframework.checker.units.qual.A;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;

public class  signLanguageClass {
    // should start from small letter

    // this is used to load model and predict
    private Interpreter interpreter;

    private Interpreter interpreter2;
    // store all label in array
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3; // for RGB
    private int IMAGE_MEAN=0;
    private  float IMAGE_STD=255.0f;
    // use to initialize gpu in app
    private GpuDelegate gpuDelegate;
    private int height=0;
    private  int width=0;
    private int Classsification_Input_Size = 0;

    private String final_text="";
    private String current_text="";

    private TextToSpeech textToSpeech;

    signLanguageClass (Context context,Button clear_button, Button add_button, TextView change_text,Button text_speech_button, AssetManager assetManager, String modelPath, String labelPath, int inputSize, String classification_mode, int classification_inputsize) throws IOException{
        INPUT_SIZE=inputSize;
        Classsification_Input_Size=classification_inputsize;
        // use to define gpu
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        options.setNumThreads(4);
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        labelList=loadLabelList(assetManager,labelPath);
        //Here***************************
        Interpreter.Options option2 = new Interpreter.Options();
        option2.setNumThreads(3);
        interpreter2=new Interpreter(loadModelFile(assetManager,classification_mode),option2);

        clear_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if(final_text.length()!=0){
                final_text = final_text.substring(0, final_text.length() - 1);
                change_text.setText(final_text);}
            }
        });

        add_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                final_text = final_text+current_text;
                change_text.setText(final_text);
            }
        });

        textToSpeech = new TextToSpeech(context, new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {
                if(status!=TextToSpeech.ERROR){
                    textToSpeech.setLanguage(Locale.US);
                }
            }
        });

        text_speech_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                textToSpeech.speak(final_text,TextToSpeech.QUEUE_FLUSH,null);

            }
        });



    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        // to store label
        List<String> labelList=new ArrayList<>();
        // create a new reader
        BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        // loop through each line and store it to labelList
        while ((line=reader.readLine())!=null){
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // use to get description of file
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset =fileDescriptor.getStartOffset();
        long declaredLength=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    // create new Mat function
    public Mat recognizeImage(Mat mat_image){

        Mat rotated_mat_image=new Mat();

        Mat a=mat_image.t();
        Core.flip(a,rotated_mat_image,1);
        a.release();

        Bitmap bitmap=null;
        bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);
        height=bitmap.getHeight();
        width=bitmap.getWidth();
        Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);
        Object[] input=new Object[1];
        input[0]=byteBuffer;

        Map<Integer,Object> output_map=new TreeMap<>();

        float[][][]boxes =new float[1][10][4];
        float[][] scores=new float[1][10];
        float[][] classes=new float[1][10];

        // add it to object_map;
        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        // now predict
        interpreter.runForMultipleInputsOutputs(input,output_map);
        Object value=output_map.get(0);
        Object Object_class=output_map.get(1);
        Object score=output_map.get(2);
        for (int i=0;i<10;i++){
            float class_value=(float) Array.get(Array.get(Object_class,0),i);
            float score_value=(float) Array.get(Array.get(score,0),i);

            if(score_value>0.5){
                Object box1=Array.get(Array.get(value,0),i);
                float y1=(float) Array.get(box1,0)*height;
                float x1=(float) Array.get(box1,1)*width;
                float y2=(float) Array.get(box1,2)*height;
                float x2=(float) Array.get(box1,3)*width;

                if (y1<0){
                    y1 = 0;
                }
                if(x1 < 0){
                    x1 = 0;
                }
                if(x2>width){
                    x2 = width;
                }
                if(y2>height){
                    y2 = height;
                }
                float w1 = x2-x1;
                float h1 = y2-y1;

                Rect  cropped_roi = new Rect((int)x1,(int)y1,(int)w1,(int)h1);
                Mat cropped = new Mat(rotated_mat_image,cropped_roi).clone();
                Bitmap bitmap1 = null;
                bitmap1 = Bitmap.createBitmap(cropped.cols(),cropped.rows(),Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped,bitmap1);

                Bitmap scaledBitmap1 = Bitmap.createScaledBitmap(bitmap1,Classsification_Input_Size,Classsification_Input_Size,false);
                ByteBuffer byteBuffer1=convertBitmapToByteBuffer1(scaledBitmap1);

                float[][] output_class_value = new float[1][1];
                float[][] output_class_value2 = new float[1][1];

                interpreter2.run(byteBuffer1,output_class_value);
                interpreter2.run(byteBuffer1,output_class_value2);

                Log.d("signLanguageClass","output_class_value"+output_class_value[0][0]);
                Log.d("signLanguageClass","output_class_value2"+output_class_value2[0][0]);

                String sign_val=get_alphabets(output_class_value[0][0],output_class_value2[0][0]);

                current_text = sign_val;


                Imgproc.putText(rotated_mat_image,""+sign_val,new Point(x1+10,y1+40),2,1.5,new Scalar(3, 218, 197, 255),2);
                Imgproc.rectangle(rotated_mat_image,new Point(x1,y1),new Point(x2,y2),new Scalar(0, 255, 0, 255),2);



            }

        }

        Mat b=rotated_mat_image.t();
        Core.flip(b,mat_image,0);
        b.release();
        // Now for second change go to CameraBridgeViewBase
        return mat_image;
    }

    private String get_alphabets(float sig_v,float sig_v2) {
        String val  = "";
        if((sig_v>=-0.5 & sig_v<0.5) & (sig_v2>=-0.5 & sig_v2<0.5)){
            val = "A";
        }else if((sig_v>=0.5 & sig_v<1.5)& (sig_v2>=0.5 & sig_v2<1.5)){
            val = "B";
        }else if((sig_v>=1.5 & sig_v<2.5)& (sig_v2>=1.5 & sig_v2<2.5)){
            val = "C";
        }else if((sig_v>=2.5 & sig_v<3.5)& (sig_v2>=2.5 & sig_v2<3.5)){
            val = "D";
        }else if((sig_v>=3.5 & sig_v<4.5)& (sig_v2>=3.5 & sig_v2<4.5)){
            val = "E";
        }else if((sig_v>=4.5 & sig_v<5.5)& (sig_v2>=4.5 & sig_v2<5.5)){
            val = "F";
        }else if((sig_v>=5.5 & sig_v<6.5)& (sig_v2>=5.5 & sig_v2<6.5)){
            val = "G";
        }else if((sig_v>=6.5 & sig_v<7.5)& (sig_v2>=6.5 & sig_v2<7.5)){
            val = "H";
        }else if((sig_v>=7.5 & sig_v<8.5)& (sig_v2>=7.5 & sig_v2<8.5)){
            val = "I";
        }else if((sig_v>=8.5 & sig_v<9.5)& (sig_v2>=8.5 & sig_v2<9.5)){
            val = "J";
        }else if((sig_v>=9.5 & sig_v<10.5)& (sig_v2>=9.5 & sig_v2<10.5)){
            val = "K";
        }else if((sig_v>=10.5 & sig_v<11.5)& (sig_v2>=10.5 & sig_v2<11.5)){
            val = "L";
        }else if((sig_v>=11.5 & sig_v<12.5)& (sig_v2>=11.5 & sig_v2<12.5)){
            val = "M";
        }else if((sig_v>=12.5 & sig_v<13.5)& (sig_v2>=12.5 & sig_v2<13.5)){
            val = "N";
        }else if((sig_v>=13.5 & sig_v<14.5)& (sig_v2>=13.5 & sig_v2<14.5)){
            val = "O";
        }else if((sig_v>=14.5 & sig_v<15.5)& (sig_v2>=14.5 & sig_v2<15.5)){
            val = "P";
        }else if((sig_v>=15.5 & sig_v<16.5)& (sig_v2>=15.5 & sig_v2<16.5)){
            val = "Q";
        }else if((sig_v>=16.5 & sig_v<17.5)& (sig_v2>=16.5 & sig_v2<17.5)){
            val = "R";
        }else if((sig_v>=17.5 & sig_v<18.5)& (sig_v2>=17.5 & sig_v2<18.5)){
            val = "S";
        }else if((sig_v>=18.5 & sig_v<19.5)& (sig_v2>=18.5 & sig_v2<19.5)){
            val = "T";
        }else if((sig_v>=19.5 & sig_v<20.5)& (sig_v2>=19.5 & sig_v2<20.5)){
            val = "U";
        }else if((sig_v>=20.5 & sig_v<21.5)& (sig_v2>=20.5 & sig_v2<21.5)){
            val = "V";
        }else if((sig_v>=21.5 & sig_v<22.5)& (sig_v2>=21.5 & sig_v2<22.5)){
            val = "W";
        }else if((sig_v>=22.5 & sig_v<23.5)& (sig_v2>=22.5 & sig_v2<23.5)){
            val = "X";
        }else if((sig_v>=23.5 & sig_v<24.5)& (sig_v2>=23.5 & sig_v2<24.5)){
            val = "Y";
        }else if((sig_v>=24.5 & sig_v<25.5)& (sig_v2>=24.5 & sig_v2<25.5)){
            val = "Z";
        }else if((sig_v>=25.5 & sig_v<26.5)& (sig_v2>=25.5 & sig_v2<26.5)){
            val = " ";
        }
        return val;
    }    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int quant=1;
        int size_images=INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // some error
        //now run
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    // paste this
                    byteBuffer.putFloat((((val >> 16) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val >> 8) & 0xFF))/255.0f);
                    byteBuffer.putFloat((((val) & 0xFF))/255.0f);
                }
            }
        }
        return byteBuffer;
    }

    private ByteBuffer convertBitmapToByteBuffer1(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int quant=1;
        int size_images=Classsification_Input_Size;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else {
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel=0;

        // some error
        //now run
        for (int i=0;i<size_images;++i){
            for (int j=0;j<size_images;++j){
                final  int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else {
                    // paste this
                    byteBuffer.putFloat((((val >> 16) & 0xFF)));
                    byteBuffer.putFloat((((val >> 8) & 0xFF)));
                    byteBuffer.putFloat((((val) & 0xFF)));
                }
            }
        }
        return byteBuffer;
    }
}