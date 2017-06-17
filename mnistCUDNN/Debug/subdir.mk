################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../fp16_emu.cpp \
../mnistCUDNN.cpp 

CU_SRCS += \
../fp16_dev.cu 

CU_DEPS += \
./fp16_dev.d 

OBJS += \
./fp16_dev.o \
./fp16_emu.o \
./mnistCUDNN.o 

CPP_DEPS += \
./fp16_emu.d \
./mnistCUDNN.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/yildbs/Workspace/cuDNN/Workspace/mnistCUDNN/FreeImage/include" -G -g -O0 -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/yildbs/Workspace/cuDNN/Workspace/mnistCUDNN/FreeImage/include" -G -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/yildbs/Workspace/cuDNN/Workspace/mnistCUDNN/FreeImage/include" -G -g -O0 -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/yildbs/Workspace/cuDNN/Workspace/mnistCUDNN/FreeImage/include" -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


