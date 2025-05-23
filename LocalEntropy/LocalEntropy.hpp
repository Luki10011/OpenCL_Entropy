/**********************************************************************
Copyright �2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#ifndef SOBEL_FILTER_IMAGE_H_
#define SOBEL_FILTER_IMAGE_H_

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>   
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

#include <iostream>
#include <vector>
#include <filesystem>
namespace fs = std::filesystem;


#define SAMPLE_VERSION "AMD-APP-SDK-v3.0.130.1"

#define INPUT_DIR "input/"  // Path to input images
#define OUTPUT_DIR "output/"  // Path to generate output images
#define INPUT_IMAGE "LocalEntropy_Input"
#define OUTPUT_IMAGE "LocalEntropy_Output"
#define DEF_STRUCT "square"
#define DEF_M 5
#define DEF_N 5

#define GROUP_SIZE 256

using namespace appsdk;

std::vector<std::string> findBmpFiles(const std::string& folderPath);

/**
* @class LocalEntropy
* Klasa implementująca algorytm obliczania lokalnej entropii dla obrazu
*/
class LocalEntropy
{
    cl_double setupTime;                /**< czas potrzebny na przygotowanie zasobów OpenCL i kompilację kernela */
    cl_double kernelTime;               /**< czas potrzebny na uruchomienie kernela i odczytanie wyników */
    cl_uchar4* inputImageData;          /**< dane wejściowe bitmapy przesyłane do urządzenia */
    cl_uchar4* outputImageData;         /**< wynik otrzymany z urządzenia */

    cl::Context context;                            /**< kontekst OpenCL */
    std::vector<cl::Device> devices;                /**< lista urządzeń OpenCL */
    std::vector<cl::Device> device;                 /**< używane urządzenie OpenCL */
    std::vector<cl::Platform> platforms;            /**< lista platform OpenCL */
    cl::Image2D inputImage2DGray;                   /**< wejściowy obraz 2D (w skali szarości) */
    cl::Image2D outputImage2DGray;                  /**< wyjściowy obraz 2D (w skali szarości) */
    cl::Image2D inputImage2DEntropy;                /**< wejściowy obraz 2D dla obliczeń entropii */
    cl::Image2D outputImage2DEntropy;               /**< wyjściowy obraz 2D dla obliczeń entropii */
    cl::CommandQueue commandQueue;                  /**< kolejka poleceń OpenCL */
    cl::Program program;                            /**< program OpenCL */
    cl::Kernel grayscaleKernel;                     /**< kernel do konwersji na skalę szarości */
    cl::Kernel entropyKernel;                       /**< kernel do obliczania lokalnej entropii */

    cl_uchar* verificationOutput;       /**< tablica wyników dla implementacji referencyjnej */

    SDKBitMap inputBitmap;              /**< obiekt klasy bitmapy */
    uchar4* pixelData;                  /**< wskaźnik na dane obrazu */
    cl_uint pixelSize;                  /**< rozmiar piksela w formacie BMP */
    cl_uint width;                      /**< szerokość obrazu */
    cl_uint height;                     /**< wysokość obrazu */
    cl_bool byteRWSupport;
    size_t kernelWorkGroupSize;         /**< rozmiar grupy roboczej zwrócony przez kernel */
    size_t blockSizeX;                  /**< rozmiar grupy roboczej w kierunku X */
    size_t blockSizeY;                  /**< rozmiar grupy roboczej w kierunku Y */
    int iterations;                     /**< liczba iteracji wykonania kernela */
    int imageSupport;
    std::string inputName;              /**< nazwa pliku wejściowego obrazu */
    std::string structShape;            /**< kształt elementu strukturalnego */
    std::string input_type;             /**< typ wejścia [folder lub obraz] */
    int mStruct;                        /**< liczba wierszy elementu strukturalnego */
    int nStruct;                        /**< liczba kolumn elementu strukturalnego */
    std::string Alt_ImageName;
    std::vector<cl_uchar> structElem;   /**< spłaszczony element strukturalny do filtracji entropii lokalnej */
    int is_dir;                         /**< flaga określająca, czy wejściem jest folder, czy obraz */
    int print_flag;
    SDKTimer* sampleTimer;              /**< obiekt klasy SDKTimer */
    float totalTime_inSeconds;          /**< czas całkowity: przygotowanie + wykonanie kernela */
    float totalKernelTime_inSeconds;    /**< całkowity czas wykonania kernela */

public:

    CLCommandArgs* sampleArgs;          /**< klasa obsługująca argumenty wejściowe CL */

        int readInputImage(std::string inputImageName);

        int writeOutputImage(std::string outputImageName);

        /**
        * Konstruktor
        * Inicjalizuje parametry obiektu
        */
        LocalEntropy()
            : inputImageData(NULL),
              outputImageData(NULL),
              verificationOutput(NULL),
              byteRWSupport(true)
        {
            sampleArgs = new CLCommandArgs();
            sampleTimer = new SDKTimer();
            sampleArgs->sampleVerStr = SAMPLE_VERSION;
            pixelSize = sizeof(uchar4);
            pixelData = NULL;
            blockSizeX = GROUP_SIZE;
            blockSizeY = 1;
            iterations = 1;
            imageSupport = 0;
            inputName = INPUT_IMAGE;
            structShape = DEF_STRUCT;
            mStruct = 0;
            nStruct = 0;
            Alt_ImageName = "default";
            is_dir = 0;
            print_flag = 1;
            totalTime_inSeconds = 0;
            totalKernelTime_inSeconds = 0;
        }

        LocalEntropy(std::string input_img_name, int print_flag)
        : inputImageData(NULL),
          outputImageData(NULL),
          verificationOutput(NULL),
          byteRWSupport(true)
    {
        sampleArgs = new CLCommandArgs();
        sampleTimer = new SDKTimer();
        sampleArgs->sampleVerStr = SAMPLE_VERSION;
        pixelSize = sizeof(uchar4);
        pixelData = NULL;
        blockSizeX = GROUP_SIZE;
        blockSizeY = 1;
        iterations = 1;
        imageSupport = 0;
        inputName = INPUT_IMAGE;
        structShape = DEF_STRUCT;
        mStruct = 0;
        nStruct = 0;
        Alt_ImageName = input_img_name;
        is_dir = 0;
        print_flag = print_flag;
        totalTime_inSeconds = 0;
        totalKernelTime_inSeconds = 0;
    }

        ~LocalEntropy()
        {
        }

        int setupLocalEntropy();

        std::string getInputName(){
            return inputName;
        }

        std::string getInputType(){
            return input_type;
        }

        void setInputName(std::string name){
            inputName = name;
        }

        int genBinaryImage();

        int setupCL();

        int runCLKernels();

        void printStats();

        int initialize();

        int setup();

        int readInputImage_wrapper();

        int run();

        int cleanup();

        std::string getAltImgName(){
            return Alt_ImageName;
        }

        int get_is_dir(){
            return is_dir;
        }

        float getTotalTime(){
            return totalTime_inSeconds;
        }

        float getTotalKernelTime(){
            return totalKernelTime_inSeconds;
        }


};

#endif // SOBEL_FILTER_IMAGE_H_
