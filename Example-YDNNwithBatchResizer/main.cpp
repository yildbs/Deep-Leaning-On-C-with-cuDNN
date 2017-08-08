#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include <ydnn/YNetwork.h>
#include <ycuda/resizer/YCudaBatchResizer.h>

using namespace ydnn;

std::vector<std::string> GetKISALabelString(){
	std::vector<std::string> labels;
	labels.push_back("Not human");
	labels.push_back("Human");
	return labels;
}

#define IMSHOW_RESIZER 0

int main()
{
	std::cout << "Hello !" << std::endl;

	// Initialize YLeNet
	YLeNet cnn;
	cnn.Initialize();

	//cv::Mat frame = cv::imread("../../Auxiliaries/frame.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat frame = cv::imread("../../Auxiliaries/frame.jpg");

	const size_t shape[] = {128, 128, 3};
	auto start_total = std::chrono::high_resolution_clock::now();

	// Initialize YCudaBatchResizer
	// Source image : GRAY
	// Destination image : GRAY
	ycuda::resizer::YCudaBatchResizer resizer;
	resizer
		.SetSourceSize(frame.cols, frame.rows, ycuda::resizer::YCudaBatchResizer::GRAY)
		.SetDestinationSize(shape[0], shape[1], ycuda::resizer::YCudaBatchResizer::GRAY);

	// Initialize input tensor
	auto labels = GetKISALabelString();
	size_t num_labels = labels.size();
	const size_t batch_size = 100;
	YTensor input = YTensor().SetTensorDescriptor(batch_size, shape[1], shape[0], shape[2], false);

	// Rect buffer
	std::vector<cv::Rect> rects;
	rects.push_back(cv::Rect(667, 337, 222, 222));
	rects.push_back(cv::Rect(938, 393, 222, 222));
	rects.push_back(cv::Rect(0, 100, 330, 330));
	rects.push_back(cv::Rect(300, 500, 100, 100));
	rects.push_back(cv::Rect(600, 100, 100, 100));

	// Set location of objects
	int num_rects = 10;
	resizer.SetNumMatrix(num_rects);
	resizer.ResetRects();
	for (auto rect : rects){
		resizer.PushRect(rect.x, rect.y, rect.width, rect.height);
	}

	// Resize!
	int num_resized_images = resizer.CudaBatchResize(frame.rows*frame.cols*frame.channels(), frame.data);
	std::cout << "num_resized_images : " << num_resized_images << std::endl;

#if IMSHOW_RESIZER
	cv::Mat resized[10];
	float* ptr = resizer.GetDstBits();
	for(int img_idx=0;img_idx<num_resized_images;img_idx++){
		resized[img_idx] = cv::Mat(shape[1], shape[0], shape[2] == 1 ? CV_8UC1 : CV_8UC3);
		for(int idx=0;idx<shape[0]*shape[1]*shape[2];idx++){
			float value = ptr[img_idx*shape[0]*shape[1]*shape[2]+ idx] * 255.0f;
			if( value >= 255.0f ){
				value = 255.0;
			}
			resized[img_idx].data[idx] = (unsigned char)value;
		}
		char frame_name[100];
		std::string img_name = "resized " + std::to_string(img_idx);
		cv::imshow(img_name.c_str(), resized[img_idx]);
	}
#endif

	// Predict
	input.SetBatchSize(num_resized_images);
	input.SetData(resizer.GetDst());

	auto start_cnn_process = std::chrono::high_resolution_clock::now();
	auto softmax_output = cnn.Process(input);
	std::chrono::duration<double> elapsed_cnn_process = std::chrono::high_resolution_clock::now() - start_cnn_process;
	std::chrono::duration<double> elapsed_total = std::chrono::high_resolution_clock::now() - start_total;
	std::cout << "CNN Process elapsed time ! : " << elapsed_cnn_process.count() << std::endl;
	std::cout << "Total elapsed time ! : " << elapsed_total.count() << std::endl;

	// Draw rects	
	for (int idx = 0; idx < rects.size(); idx++){
		auto rect = rects[idx];
		printf("---------------%d\n", idx);

		for (int label = 0; label < num_labels; label++){
			printf("%2d. %.5f. %s\n", label, softmax_output[idx*num_labels+label], labels[label].c_str());
		}
		auto prob_human = softmax_output[idx*num_labels + 1];
		bool is_human = prob_human > 0.95;

		cv::Scalar colors[2];
		colors[0] = cv::Scalar(255, 0, 0);
		colors[1] = cv::Scalar(0, 0, 255);
		auto color = is_human ? colors[0] : colors[1];

		cv::rectangle(frame, rect, color, 5);
		char text[30];
		if (is_human){
			sprintf(text, "%d. Human", idx);
		}
		else{
			sprintf(text, "%d. Not human", idx);
		}
		cv::putText(frame, std::string(text), cv::Point(rect.x, rect.y), 1, 3, color, 5);
	}
	cv::imshow("frame", frame);
	cv::waitKey(0);
}
