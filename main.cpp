
/*  Sample RUN : g++ main.cpp -o output `pkg-config --cflags --libs opencv`
                 ./output
*/


#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include<bits/stdc++.h>

using namespace cv;
using namespace std;


 float PSNR(Mat image1,Mat image2){
 float mse = 0;
  for(int i=1;i<image1.rows;i++){
    for(int j=1;j<image1.cols;j++){
       float diff = image1.at<uchar>(i,j) - image2.at<uchar>(i,j);
       diff = diff*diff;
       mse = mse + diff;
    
    
    }
  
  }
  mse = mse/(image1.rows*image1.cols);
  
  float psnr = 10*(log((255*255)/mse));
  return psnr;
 
 
 }

  int log_transformation(Mat image)
{
   
 int dest_w=(image.cols);
 int dest_h=(image.rows);
   Mat image2(dest_h,dest_w,CV_8UC3,Scalar(0));
   for(int i=0;i<dest_h;i++)
{  

   for(int j=0;j<dest_w;j++)
   {
   
   image.at<uchar>(i,j) =  38*log(1+(image.at<uchar>(i,j))); 
 
   }
}

 // namedWindow("Source image", CV_WINDOW_AUTOSIZE);
    namedWindow("log_transformed image",WINDOW_AUTOSIZE);
  //  imshow("Source image", image);
    imshow("log_transformed image", image);
   imwrite( "log_transformed_image.jpg",image );
 waitKey(0);
 return 0;
}


int scaling(Mat image){

cout<<"Original width and height of image are "<<image.cols<<" and "<<image.rows<<endl;
cout<<"Enter the scaling factor in x and y direction"<<endl;
 float dest_w,dest_h;
  float sfx,sfy;
 
  cin>>sfx>>sfy; 
 dest_h=image.rows*sfx;
 dest_w=image.cols*sfy;

 
  Mat image2(dest_h,dest_w,CV_8UC1,Scalar(0));
  int tempx=0,tempy=0;

   for(float i=0;i<dest_h;i++)
{  
    tempx = (int)(((1/sfx)*(i-0.5))+0.5);

   for(float j=0;j<dest_w;j++)
   {

   tempy = (int)(((1/sfy)*(j-0.5))+0.5);

   image2.at<uchar>(i,j) =  image.at<uchar>(tempx,tempy); 
 // image2.at<Vec3b>(i,j) =  image.at<Vec3b>(tempx,tempy); 
  

   }
}
  
  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
  
 namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
 imshow( "Display window", image2 );
  imwrite( "scaling_image.jpg",image2 );
 waitKey(0);
return 0;
}

int negative(Mat image)
{
   
double dest_w=(image.cols);
double dest_h=(image.rows);
    Mat image2(dest_h,dest_w,CV_8UC1,Scalar(0));
  int tempx=0,tempy=0;

   for(float i=0;i<dest_h;i++)
{  

   for(float j=0;j<dest_w;j++)
   {
     
  image2.at<uchar>(i,j) =  255 - image.at<uchar>(i,j);

   }
}

  namedWindow( "Display window",WINDOW_AUTOSIZE );
 imshow( "Display window", image2 );
 imwrite( "negative_image.jpg",image2 );
  
 waitKey(0);
 return 1;;
}
int resize_nearest_neighbour(Mat image){

cout<<"Original width and height of image are "<<image.cols<<" and "<<image.rows<<endl;
cout<<"Enter the new width and height of image"<<endl;
 float dest_w,dest_h;
  float sfx,sfy;
 
  cin>>dest_w>>dest_h; 
 sfx=dest_h/image.rows;
 sfy=dest_w/image.cols;

 
  Mat image2(dest_h,dest_w,CV_8UC1,Scalar(0));
   Mat inbuiltimg(dest_h,dest_w,CV_8UC1,Scalar(0));
  int tempx=0,tempy=0;

   for(float i=0;i<dest_h;i++)
{  
    tempx = (int)(((1/sfx)*(i-0.5))+0.5);

   for(float j=0;j<dest_w;j++)
   {

   tempy = (int)(((1/sfy)*(j-0.5))+0.5);

   image2.at<uchar>(i,j) =  image.at<uchar>(tempx,tempy); 
 // image2.at<Vec3b>(i,j) =  image.at<Vec3b>(tempx,tempy); 
   
   
   }
}
  
  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
  resize(image,inbuiltimg,Size(dest_h,dest_w),0,0,INTER_NEAREST);
  cout<<"PSNR = "<<PSNR(image2,inbuiltimg);
 namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
 imshow( "Display window", image2 );
  imwrite( "resize_nearest_neighbour.jpg",image2 );
 waitKey(0);
return 0;
}
 
 
 
int resize_interpolation(Mat image){

cout<<"height and width of original image is "<<image.rows<<" and "<<image.cols<<endl;
cout<<"Enter the height and weight of final image"<<endl; 
 
float dest_w;
float dest_h;

  cin>>dest_w>>dest_h;
  
  Mat image2(dest_h,dest_w,CV_8UC1,Scalar(0));
   Mat inbuiltimg(dest_h,dest_w,CV_8UC1,Scalar(0));
  float tempx,tempy;
  float row_ratio = (image.rows)/dest_h;
  float col_ratio = (image.cols)/dest_w;
  
  int r,c;
  float delr,delc;
  float  intensity1,intensity2,intensity3,intensity4;
   for(float i=1;i<dest_h;i++)
{  
      tempx = i*row_ratio;
       r = floor(tempx);
   //  cout<<"r="<<r<<endl;
   for(float j=1;j<dest_w;j++)
   {
         tempy = j*col_ratio;
         c = floor(tempy);
         
         delr = tempx-r;
         delc = tempy-c;
         image2.at<uchar>(round(i),round(j)) = ( image.at<uchar>(r,c)*(1-delr)*(1-delc)) + (image.at<uchar>(r,c+1)*(1-delr)*(delc)) + (image.at<uchar>(r+1,c)*(delr)*(1-delc)) +     (image.at<uchar>(r+1,c+1)*(delr)*(delc));
       
   }
}
  cout<<image.at<uchar>(334,456);
  if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
   resize(image,inbuiltimg,Size(dest_h,dest_w),0,0,INTER_LINEAR);
   cout<<"PSNR = "<<PSNR(image2,inbuiltimg);
 namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
 imshow( "Display window", image2 );
 imwrite( "resize_interpolation.jpg",image2 );
  
 waitKey(0);
return 0;
}

int rotation(Mat image){
   cout<<"Enter the angle to be rotated"<<endl;
   float angle;
   cin>>angle;
    float radians = (angle * 3.14) / 180;
    	
    Mat image2(image.rows,image.cols,CV_8UC1,Scalar(0));
  // int maxx=(image.rows*cos(-radians))-(image.cols*sin(-radians));
    // rotation
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
        
        
        int width=image.cols;
        int height=image.rows;
         
    float x_center = (image.rows) / 2;
    float y_center = (image.cols) / 2;
		
		int tempx = i - x_center;
		int tempy = j - y_center;
			
		int xx = (int)round((cos(-radians) * tempx - sin(-radians) * tempy) + x_center);
		int yy = (int)round((sin(-radians) * tempx + cos(-radians) * tempy) + y_center);

		if(xx >= 0 && xx < image.cols && yy >= 0 && yy < image.rows) {
	
		 
         image2.at<uchar>(i,j) = image.at<uchar>(xx,yy);
           
}
       else{
        image2.at<uchar>(i,j) = 0;
               
       }
}

}

if(! image.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
  
                          
   
    Mat dst;
    Point2f pt(image.cols/2., image.rows/2.);    
    Mat r = getRotationMatrix2D(pt, 30, 1.0);
    warpAffine(image, dst, r, Size(image.cols, image.rows));
    

 namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
 imshow( "Rotation_without inbuilt function", image2);
 imwrite( "rotation.jpg",image2 );
 imwrite( "rotation_inbuilt.jpg",dst);
   cout<<"PSNR = "<<PSNR(image2,dst);
 waitKey(0);
return 0;
}


int translation(Mat image){

 float dest_w=(image.cols);
 float dest_h=(image.rows);
 int shiftx,shifty;
 cout<<"Enter no. of units to be translated in x and y direction"<<endl;
 cin>>shiftx>>shifty;
 dest_w+=shifty;
 dest_h+=shiftx;
   Mat image2(dest_h,dest_w,CV_8UC1,Scalar(0));
  int tempx=0,tempy=0;
  
  
   for(float i=0;i<shiftx;i++)
{  

   for(float j=0;j<shifty;j++)
   {
  
    image2.at<uchar>(i,j)  = 0;
  }
  }
   for(float i=shiftx;i<dest_h;i++)
{  

   for(float j=shifty;j<dest_w;j++)
   {
   
   
   image2.at<uchar>(i,j) =   image.at<uchar>(i-shiftx,j-shifty);
 
   }
}

  namedWindow( "Display window",WINDOW_AUTOSIZE );
 imshow( "Display window", image2 );
  imwrite( "translation.jpg",image2 );
 waitKey(0);
 return 1;

}

int histogram_equalisation(Mat image){

 Mat image2(image.rows,image.cols,CV_8UC1,Scalar(0));

 float histo[256];
 for(int d=0;d<256;d++)
 {
    histo[d]=0;
 
 }
 
 for (int i=0;i<image.rows;i++)
 {
   for (int j=0;j<image.cols;j++)
   {
     int temp = image.at<uchar>(i,j);
     histo[temp]++;
   }
   
 }
 float cumulative[256];
 cumulative[0] = histo[0];
 
 for(int k=1;k<256;k++)
 {
 cumulative[k] = histo[k] + cumulative[k-1];
 
 }

for(int i=0;i<image.rows;i++)
 {
   for (int j=0;j<image.cols;j++)
   {
       image2.at<uchar>(i,j) =(cumulative[(int)image.at<uchar>(i,j)]/(image.rows*image.cols))*255;
 }
 }
 Mat dst;
 equalizeHist(image,dst);
  cout<<"PSNR = "<<PSNR(image2,dst)<<endl;
  namedWindow("Source image", CV_WINDOW_AUTOSIZE);
    namedWindow("Histogram_equalised image",WINDOW_FREERATIO);
    imshow("Source image", image);
    imshow("Histogram_equalised image", image2);
  imwrite( "histogram_equalisation.jpg",image2 );
 waitKey(0);
 return 1;
}

int bitPlaneSlice(Mat image){
          
         
 int dest_w=(image.cols);
 int dest_h=(image.rows);
 int plane;
 cout<<"Enter the no. of bit plane slicing (say 8)"<<endl;
 cin>>plane;
 Mat image2(dest_h,dest_w,CV_8UC1,Scalar(0));
  
  int num = pow(2,plane-1);
  cout<<"num="<<num;
  for(int i=0;i<image.rows;i++)
 {
   for (int j=0;j<image.cols;j++)
   {
    int intensity = image.at<uchar>(i,j);
     int check = (intensity & num);
     if(check == 0){
       image2.at<uchar>(i,j) = 0;
 }
  
     else{
       image2.at<uchar>(i,j) = num;
     
     }   
 }
 }   
 namedWindow( "Display window",WINDOW_AUTOSIZE );
 imshow( "Display window", image2 );
  imwrite( "bit_plane.jpg",image2 );
 waitKey(0);
 return 1;

}
                                                                                                                                                               


int shear(Mat image){
  float a=0.8;
int dest_w=(image.cols);
 int dest_h=(image.rows);
 float shearx,sheary;
 cout<<"Enter the amount of shear in horizontal and vertical"<<endl;
  cin>>sheary>>shearx;
 int x1, y1; // tranformed point
    
    int maxX = (dest_w * shearx);
    int maxY = (dest_h * sheary);

  Mat image2 (dest_h + maxX, dest_w + maxY,CV_8UC1,Scalar(0));
    for (int x = 0; x < image2.rows; x++)
    {
        for (int y = 0; y < image2.cols; y++)
        {
            x1 = x + (y * shearx) - maxX;
            y1 = y+ (x * sheary)  - maxY;
           // cout<<"x1="<<x1<<"y1"<<y1<<endl;

            if (x1 >= 0 && x1 <= image.rows && y1 >= 0 && y1 <= image.cols) 
            {
                image2.at<uchar>(x, y) = image.at<uchar>(x1, y1); 
             // cout<<"xorg="<<x<<"yorg="<<y<<"xnew="<<x1<<"ynew="<<y1<<endl;
            }
            

        }
    }
 
    namedWindow("Source image", CV_WINDOW_AUTOSIZE);
    namedWindow("Rotated image",WINDOW_FREERATIO);
    imshow("Source image", image);
    imshow("sheared image", image2);
  
 imwrite( "sheared_image.jpg",image2 );
  
    waitKey(0);

    return 0;
}

int gamma(Mat image)
{
  
  int dest_w=(image.cols);
 int dest_h=(image.rows);
 
 Mat image2(dest_h,dest_w,CV_8UC1,Scalar(0));
   for(int i=0;i<dest_h;i++)
  {
    for(int j=0;j<dest_w;j++)
  {
    image2.at<uchar>(i,j) = 39*(pow(image.at<uchar>(i,j),0.3));

}
}
    namedWindow("Source image", CV_WINDOW_AUTOSIZE);
    namedWindow("Gamma_transformed image",WINDOW_FREERATIO);
    imshow("Source image", image);
    imshow("Gamma_transformed image", image2);
  imwrite( "gamma.jpg",image2 );
 waitKey(0);
 return 1;
 }



int histogram_matching(Mat image1,Mat image2){

 Mat image3(image2.rows,image2.cols,CV_8UC1,Scalar(0));

 float histo1[256];
 for(int d=0;d<256;d++)
 {
    histo1[d]=0;
 
 }
 
 for (int i=0;i<image1.rows;i++)
 {
   for (int j=0;j<image1.cols;j++)
   {
     int temp = image1.at<uchar>(i,j);
     histo1[temp]++;
   }
   }
   float pd1[256];
     
   for(int d=0;d<256;d++)
 {
    pd1[d] = histo1[d]/(image1.rows*image1.cols);
 
 }
 float cpd1[256];
 cpd1[0]=pd1[0];
 
   for(int k=1;k<256;k++)
 {
 cpd1[k] = pd1[k] + cpd1[k-1];
 
 }
   int g1[256];
    for(int d=0;d<256;d++)
 {
    g1[d] = cpd1[d]*255;
 
 }
 
 
  float histo2[256];
 for(int d=0;d<256;d++)
 {
    histo2[d]=0;
 
 }
 
 for (int i=0;i<image2.rows;i++)
 {
   for (int j=0;j<image2.cols;j++)
   {
     int temp = image2.at<uchar>(i,j);
     histo2[temp]++;
   }
   }
   float pd2[256];
   for(int d=0;d<256;d++)
 {
    pd2[d] = histo2[d]/(image2.rows*image2.cols);
 
 }
 
 float cpd2[256];
 cpd2[0]=pd2[0];
 
   for(int k=1;k<256;k++)
 {
 cpd2[k] = pd2[k] + cpd2[k-1];
 
 }
   int g2[256];
    for(int d=0;d<256;d++)
 {
    g2[d] = cpd2[d]*255;
 
 }
 
 for(int i=0;i<image2.rows;i++){
   for(int j=0;j<image2.cols;j++){
   
    int temp = image2.at<uchar>(i,j);
    temp=g2[temp];
    int finl;int chk = 345;
    for(int k=0;k<256;k++)
    {
      
      if(g1[k]>temp ){
      if(g1[k]-temp<chk)
      {  chk = g1[k]-temp;
         finl = k;
      
      }
      }
   
      else {
      
      if(temp-g1[k]<chk)
      {
       chk = temp-g1[k];
       finl = k;
      }
      
      
      }
       
    }
    
    image3.at<uchar>(i,j) = finl;
    
   }
 
 
 
 }
 namedWindow("Source image", CV_WINDOW_AUTOSIZE);
    namedWindow("Histogram_matched image",WINDOW_FREERATIO);
    imshow("Source image", image2);
    imshow("Histogram_matched image", image3);
    imwrite( "histogram_matching.jpg",image3 );
 waitKey(0);
 return 1;
   
 }

int tiePoint(Mat image){
    
int x1,y1,x2,y2,x3,y3,x4,y4;
int x11,y11,x22,y22,x33,y33,x44,y44;
cout<<"Enter the points of distorted image(sample points: 10 450 10 10 266 450 266 10)"<<endl;
cin>>x1>>y1>>x2>>y2>>x3>>y3>>x4>>y4;
cout<<"Enter the points of original image(sample points: 13 399 233 14 235 523 455 142)"<<endl;
cin>>x11>>y11>>x22>>y22>>x33>>y33>>x44>>y44;

float data[64] = {         x1,y1,x1*y1,1,0,0,0,0,
                           0,0,0,0,x1,y1,x1*y1,1,
                           x2,y2,x2*y2,1,0,0,0,0,
                           0,0,0,0,x2,y2,x2*y2,1,
                           x3,y3,x3*y3,1,0,0,0,0,
                           0,0,0,0,x3,y3,x3*y3,1,
                           x4,y4,x4*y4,1,0,0,0,0,
                           0,0,0,0,x4,y4,x4*y4,1};
                           
                           
           //  float data[9] = {1,2,3,4,5,6,7,8,9};              
Mat dt(Size(8,8),CV_32F,data);

cout<<dt<<endl;
float data2[8]={x11,y11,x22,y22,x33,y33,x44,y44};

Mat origin(8,1,CV_32F,data2);
cout<<origin<<endl;
Mat dtinv(8,8,CV_32F,Scalar(0));
dtinv = dt.inv();
Mat constant (8,1,CV_32F,Scalar(0));
cout<<dtinv<<endl;
constant = dtinv*origin;
cout<<constant<<endl;
 
 Mat image2(image.rows,image.cols,CV_8UC1,Scalar(0));
for(int i=0;i<image.rows;i++){
  for(int j=0;j<image.cols;j++){
  int xnew=constant.at<float>(0,0)*i + constant.at<float>(1,0)*j + constant.at<float>(2,0)*(i*j) + constant.at<float>(3,0);
  int ynew=constant.at<float>(4,0)*i + constant.at<float>(5,0)*j + constant.at<float>(6,0)*(i*j) + constant.at<float>(7,0);
  
  image2.at<uchar>(i,j) = image.at<uchar>(xnew,ynew);
  
  }
  }

  namedWindow("Distorted_sample image", CV_WINDOW_AUTOSIZE);
    namedWindow("Recontructed image",WINDOW_FREERATIO);
    imshow("Distorted_sampleimage", image);
    imshow("Recontructed image", image2);
    imwrite( "tiePoint.jpg",image2 );
  
 waitKey(0);
 return 1;
}

int adaptiveHistogram(Mat image){

 Mat image2(image.rows,image.cols,CV_8UC1,Scalar(0));

  cout<<"Enter the size of window (say 7x7)"<<endl;
  int wx;
  int wy;
  cin>>wx>>wy;
  int wi=image.cols;
  int hi=image.rows;
  cout<<wi<<hi;
  
  for(int i=0;i<=image.cols-wy;i++){
   for(int j=0;j<=image.rows-wx;j++){
                                                                   
      
 float histo[256];
 for(int d=0;d<256;d++)
 {
    histo[d]=0;
 
 }
 
 for (int a=i;a<i+wy;a++)
 {
   for (int b=j;b<j+wx;b++)
   {
     int temp = image.at<uchar>(a,b);
     histo[temp]++;
   }
   
 }
 float cumulative[256];
 cumulative[0] = histo[0];
 
 for(int k=1;k<256;k++)
 {
 cumulative[k] = histo[k] + cumulative[k-1];
 
 }
 
 
       image2.at<uchar>(i+(wx/2),j+(wy/2)) =(cumulative[(int)image.at<uchar>(i+(wx/2),j+(wy/2))]/(wx*wy))*255;
 
   }
   }
   namedWindow( "Display window",WINDOW_AUTOSIZE);
 imshow( "Display window", image2);
 imwrite( "adaptive_histo_at _smallWindow.jpg",image2 );
 waitKey(0);
 return 1;
     
 }
 
 
 
 int piece(Mat image){
    Mat image2 = image.clone();
 
    int r1, s1, r2, s2;
   cout<<"Enter r1, s1, r2, s2 : (say 45, 45, 123, 123)"<<endl;
   cin>>r1>>s1>>r2>>s2;
 
    for(int y = 0; y < image.rows; y++){
        for(int x = 0; x < image.cols; x++){         
             int rt=image.at<uchar>(y,x);
                int ans;
    if(rt>=0 && rt <= r1){
       ans = rt*(s1/r1);
    }
    
    else if(rt>r1 && rt <= r2){
        ans = ((s2 - s1)/(r2 - r1)) * (rt - r1) + s1;
    }
    
    else if(rt>r2 && rt <= 255){
       ans = ((255 - s2)/(255 - r2)) * (rt - r2) + s2;
           image2.at<uchar>(y,x) = (int)ans;
            }
        }
    }
 
    namedWindow("Piecewise_transformed", CV_WINDOW_AUTOSIZE);
    imshow("Piecewise_transformed", image2);
 imwrite( "piecewise.jpg",image2 );
    waitKey();
 
    return 0;
}

 int main( int argc, char** argv ) {

  Mat image1,image2,image3,tieimage;
  tieimage=imread("1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  image1 = imread("sample5.jpg", CV_LOAD_IMAGE_GRAYSCALE);
 // imwrite("hillary.jpg",image1);
  image2 = imread("girl.jpg", CV_LOAD_IMAGE_GRAYSCALE);
   image3 = imread("images3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
   Mat pieceimg = imread("girl.jpg", CV_LOAD_IMAGE_GRAYSCALE);
  
  cout<<"1.Image Resize using nearest neighbour\n2.Image Resize using bilineae interpolation\n3.Rotation\n4.Translation\n5.Scaling\n6.Shear Transformation\n7.Image Negatives\n8.Log Transformation\n9.Power Law/Gamma Transformation\n10.Piecewise Linear Transformation\n11.Bitplane Slicing\n12.Image Reconstruction using Tie points\n13.Histogram Equalisation\n14.Adaptive Histogram Equalisation\n15.Histogram Matching\n";
  int t;
  cin>>t;
  if(t==1){
  int a=resize_nearest_neighbour(image1);
  }
   if(t==2){
  int a=resize_interpolation(image1);
  }
   if(t==3){
  int a=rotation(image1);
  }
   if(t==4){
  int a= translation(image1);
  }
   if(t==5){
  int a=scaling(image1);
  }
   if(t==6){
  int a=shear(image1);
  }
   if(t==7){
  int a= negative(image1);
  }
   if(t==8){
  int a=log_transformation(image1);
  }
   if(t==9){
  int a=gamma(image1);
  }
   if(t==10){
  int a=piece(image3);
  }
   if(t==11){
  int a=bitPlaneSlice(image1);
  }
   if(t==12){
  int a= tiePoint(tieimage);
  }
   if(t==13){
  int a= histogram_equalisation(image3);
  }
   if(t==14){
  int a= adaptiveHistogram(image2);
  }
   if(t==15){
  int a=histogram_matching(image1,image3);
  }
  
 
     return 0;
} 
