#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>

#include "header.h"
#include "ParameterManager.hpp"
#include "MedianFilter.hpp"
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using namespace std;
using namespace cv;


/// Depth -> Point
pcl::PointCloud<pcl::PointXYZ>::Ptr Depth2Point(cv::Mat src, CameraParameter cam_p) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    for (int h = 0; h < src.rows; h++) {
        for (int w = 0; w < src.cols; w++) {

            unsigned short z_value_short;
            float z_value_float;
            z_value_short = src.at<short>(h, w);
//            z_value_float = src.at<float>(h, w);

            if (z_value_short > 0 || z_value_float >0.0) {
                Eigen::Vector3f v;
                v = Eigen::Vector3f::Zero();

                v.z() = (float)(z_value_short)/1000;

                if(v.z() == 0) continue;
                v.x() = v.z() * (w - cam_p.cx) * (1.0 / cam_p.fx);
                v.y() = v.z() * (h - cam_p.cy) * (1.0 / cam_p.fy);

                pcl::PointXYZ point_tmp;
                point_tmp.x = v.x();
                point_tmp.y = v.y();
                point_tmp.z = v.z();
                cloud->points.push_back(point_tmp);
            }
        }
    }
    return cloud;
}


/// main
///

int main(int argc, char** argv)
{
    /// cvui GUI
    string WINDOW_NAME = "capture_mode";

    ParameterManager cfg_param("/home/inaho-00/work/cpp/zense_filter/cfg/recognition_parameter.toml");

    int image_width = cfg_param.ReadIntData("Camera", "image_width");
    int image_height = cfg_param.ReadIntData("Camera", "image_height");
    string pcd_filename = cfg_param.ReadStringData("Debug", "pcd_filename");
    string pcd_flt_filename = cfg_param.ReadStringData("Debug", "pcd_flt_filename");
    
    /// default folder check
    int mkdir_e = mkdir("../capture_tmp/" , 0777);
    int file_count = -2; // "." ".."分をカウントから引く
    DIR* dp=opendir("../capture_tmp/");

    if (dp!=NULL)
    {
        struct dirent* dent;
        do{
            dent = readdir(dp);
            if (dent!=NULL){
//                cout<<dent->d_name<<endl;
                file_count++;
            }
        }while(dent!=NULL);
        closedir(dp);
    }

    file_count = file_count / 5; // depth, color, color_depth, pcd, ir
    std::cout << "file count = " << file_count << std::endl;
    CameraParameter camera_param;
    camera_param.fx = cfg_param.ReadFloatData("Debug", "camera_fx");
    camera_param.fy = cfg_param.ReadFloatData("Debug", "camera_fy");
    camera_param.cx = cfg_param.ReadFloatData("Debug", "camera_cx");
    camera_param.cy = cfg_param.ReadFloatData("Debug", "camera_cy");

    string read_depth_image = cfg_param.ReadStringData("Debug", "depth_filename");
    cv::Mat depth = cv::imread(read_depth_image, IMREAD_UNCHANGED);
    pcl::PointCloud<pcl::PointXYZ>::Ptr point(new pcl::PointCloud <pcl::PointXYZ>);
    point = Depth2Point(depth, camera_param);

    int filter_length = 9; //奇数じゃないとメジアンの実装的に落ちます
    DepthMedianFilter dfilter(filter_length, depth.cols, depth.rows);
    string read_depth_image_1 = cfg_param.ReadStringData("Debug", "depth_filename_1");
    cv::Mat depth_1 = cv::imread(read_depth_image_1, IMREAD_UNCHANGED);    
    string read_depth_image_2 = cfg_param.ReadStringData("Debug", "depth_filename_2");
    cv::Mat depth_2 = cv::imread(read_depth_image_2, IMREAD_UNCHANGED);    
    string read_depth_image_3 = cfg_param.ReadStringData("Debug", "depth_filename_3");
    cv::Mat depth_3 = cv::imread(read_depth_image_3, IMREAD_UNCHANGED);    
    string read_depth_image_4 = cfg_param.ReadStringData("Debug", "depth_filename_4");
    cv::Mat depth_4 = cv::imread(read_depth_image_4, IMREAD_UNCHANGED);    
    string read_depth_image_5 = cfg_param.ReadStringData("Debug", "depth_filename_5");
    cv::Mat depth_5 = cv::imread(read_depth_image_5, IMREAD_UNCHANGED);    
    string read_depth_image_6 = cfg_param.ReadStringData("Debug", "depth_filename_6");
    cv::Mat depth_6 = cv::imread(read_depth_image_6, IMREAD_UNCHANGED);    
    string read_depth_image_7 = cfg_param.ReadStringData("Debug", "depth_filename_7");
    cv::Mat depth_7 = cv::imread(read_depth_image_7, IMREAD_UNCHANGED);    
    string read_depth_image_8 = cfg_param.ReadStringData("Debug", "depth_filename_8");
    cv::Mat depth_8 = cv::imread(read_depth_image_8, IMREAD_UNCHANGED);    
    string read_depth_image_9 = cfg_param.ReadStringData("Debug", "depth_filename_9");
    cv::Mat depth_9 = cv::imread(read_depth_image_9, IMREAD_UNCHANGED);    
    
    dfilter.enqueue(depth_1);
    dfilter.enqueue(depth_2);
    dfilter.enqueue(depth_3);
    dfilter.enqueue(depth_4);
    dfilter.enqueue(depth_5);
    dfilter.enqueue(depth_6);
    dfilter.enqueue(depth_7);
    dfilter.enqueue(depth_8);
    dfilter.enqueue(depth_9);

    cv::Mat depth_filt = dfilter.process();
    cv::Mat conf = dfilter.get_conf();

    point->width    = point->points.size();
    point->height   = 1;
    point->is_dense = false;
    point->points.resize (point->width * point->height);

    pcl::PointCloud<pcl::PointXYZ>::Ptr point_filt(new pcl::PointCloud <pcl::PointXYZ>);
    point_filt = Depth2Point(depth_filt, camera_param);
    point_filt->width    = point_filt->points.size();
    point_filt->height   = 1;
    point_filt->is_dense = false;
    point_filt->points.resize (point_filt->width * point_filt->height);

    cv::imwrite("../data/conf.png", conf);

    pcl::io::savePCDFileASCII (pcd_filename, *point);
    pcl::io::savePCDFileASCII (pcd_flt_filename, *point_filt);
    cout << "saved:" << point->points.size() << endl;                    

    return 0;
}