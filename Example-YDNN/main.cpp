#include <iostream>
#include <chrono>
#include <random>
#include <algorithm>
#include <string>

#include <ydnn/YNetwork.h>

using namespace ydnn;

std::vector<std::string> GetKISALabelString(){
	std::vector<std::string> labels;
	labels.push_back("car");//0
	labels.push_back("fire");//1
	labels.push_back("human_bowed");//2
	labels.push_back("human_clustering");//3
	labels.push_back("human_clustering_upper_body");//4
	labels.push_back("human_fall_down");//5
	labels.push_back("human_full_body");//6
	labels.push_back("human_full_body_with_shadow");//7
	labels.push_back("human_in_water");//8
	labels.push_back("human_lower_body");//9
	labels.push_back("human_running");//10
	labels.push_back("human_sitting");//11
	labels.push_back("human_sitting_with_shadow");//12
	labels.push_back("human_upper_body");//13
	labels.push_back("human_violence_a_man");//14
	labels.push_back("human_violence_clustering");//15
	labels.push_back("human_with_car");//16
	labels.push_back("human_without_head");//17
	labels.push_back("smoke");//18
	labels.push_back("water");//19
	return labels;
}

std::vector<std::string> GetINRIALabelString(){
	std::vector<std::string> labels;
	labels.push_back("Not human");
	labels.push_back("Human");
	return labels;
}


#define USE_OPENCV 1
#if USE_OPENCV
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#endif

#define TEST_KISA 1

int main()
{
#if TEST_KISA
	std::cout << "Hello !" << std::endl;

	YLeNet cnn;
	cnn.Initialize();

#if USE_OPENCV 
	//KISA
	int total_num_images = 20000;
	std::vector<cv::Mat> images;
	std::ifstream file("../../Data/KISA/manipulated_ratio/list_test_data_path.txt");
    std::string str;
    std::vector<std::string> file_paths;
    while (std::getline(file, str)){
        // Process str
    	cv::Mat image = cv::imread("../../Data/KISA/manipulated_ratio/" + str, CV_LOAD_IMAGE_GRAYSCALE);
    	file_paths.push_back("../../Data/KISA/manipulated_ratio/" + str);
		assert(image.empty() == false);
		images.push_back(image);
		if(total_num_images-- == 0){
			break;
		}
    }

	const size_t shape[] = {128, 128, 1};

	auto start_total = std::chrono::high_resolution_clock::now();

	//Resize
	for(auto& image : images){
		cv::resize(image, image, cv::Size(shape[1], shape[0]));
	}

	int num_images = images.size();
	std::vector<std::vector<float>> fimages(num_images);
	for(int idx=0;idx<num_images;idx++){
		int h, w, c;
		cv::Mat& image = images[idx];
		h = image.rows;
		w = image.cols;
		c = image.channels();
		YUtils::MatToFloatVector(h, w, c, image.data, fimages[idx]);
	}

	////////////////////////////////////////
	//Predict
	////////////////////////////////////////
//	auto labels = GetKISALabelString();
	auto labels = GetINRIALabelString();
	size_t num_labels = labels.size();
	const size_t batch_size = 100;
	const size_t total_batch = images.size() / batch_size;
	YTensor input = YTensor().SetTensorDescriptor(batch_size, shape[0], shape[1], shape[2], false);
	for (int batch_idx = 0; batch_idx < total_batch; batch_idx++){
		int start_image_index = batch_idx*batch_size;
		int end_image_index = (batch_idx + 1)*batch_size > images.size() ? images.size() : (batch_idx + 1)*batch_size;

		//HERE
		input.SetBatchSize(batch_size);

		for(int idx=start_image_index;idx<end_image_index;idx++){
			input.CopyFrom(fimages[0].size()*(idx - start_image_index), fimages[0].size(), fimages[idx].data());
		}
		auto start_cnn_process = std::chrono::high_resolution_clock::now();
		auto softmax_output = cnn.Process(input);
		std::chrono::duration<double> elapsed_cnn_process = std::chrono::high_resolution_clock::now() - start_cnn_process;
		std::chrono::duration<double> elapsed_total = std::chrono::high_resolution_clock::now() - start_total;
		std::cout << "CNN Process elapsed time ! : " << elapsed_cnn_process.count() << std::endl;
		std::cout << "Total elapsed time ! : " << elapsed_total.count() << std::endl;

		char quit_key = 0;
		for(int idx=start_image_index;idx<end_image_index;idx++){
			std::cout << "*************************" << std::endl;
			std::cout << "file_path : " << file_paths.at(idx) << std::endl;
			std::cout << "Image index : " << idx << std::endl;
			cv::resize(images[idx], images[idx], cv::Size(250,250));
			cv::imshow("image", images[idx]);
			printf("---------------\n");
			for(int label=0;label<num_labels;label++){
				printf("%2d. %.5f. %s\n", label, softmax_output[(idx-start_image_index)*num_labels + label], labels[label].c_str());
			}
			quit_key = cv::waitKey(0);
			if(quit_key == 'q' || quit_key == 'Q'){
				break;
			}
		}
		std::cout << "Mini batch end" << std::endl;
		if(quit_key == 'q' || quit_key == 'Q'){
			break;
		}
	}
#endif
#endif

}
