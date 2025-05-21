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


#include "LocalEntropy.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>


///************************** NIE USUWAC *****************************
int
LocalEntropy::readInputImage(std::string inputImageName)
{
    // load input bitmap image
    inputBitmap.load(inputImageName.c_str());

    // error if image did not load
    if(!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!" << std::endl;
        return SDK_FAILURE;
    }
    else
    {
        std::cout << "Loaded input image: " << inputImageName << std::endl;
    }

    // get width and height of input image
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();

    // allocate memory for input & output image data  */
    inputImageData  = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

    // error check
    CHECK_ALLOCATION(inputImageData, "Failed to allocate memory! (inputImageData)");


    // allocate memory for output image data
    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));

    // error check
    CHECK_ALLOCATION(outputImageData,
                     "Failed to allocate memory! (outputImageData)");


    // initialize the Image data to NULL
    memset(outputImageData, 0, width * height * pixelSize);

    // get the pointer to pixel data
    pixelData = inputBitmap.getPixels();

    // error check
    CHECK_ALLOCATION(pixelData, "Failed to read pixel Data!");

    // Copy pixel data into inputImageData
    memcpy(inputImageData, pixelData, width * height * pixelSize);

    // allocate memory for verification output
    verificationOutput = (cl_uchar*)malloc(width * height * pixelSize);

    // error check
    CHECK_ALLOCATION(verificationOutput,
                     "verificationOutput heap allocation failed!");

    // initialize the data to NULL
    memset(verificationOutput, 0, width * height * pixelSize);

    return SDK_SUCCESS;

}
///*******************************************************************


///************************** NIE USUWAC *****************************
int
LocalEntropy::writeOutputImage(std::string outputImageName)
{   
    // copy output image data back to original pixel data
    memcpy(pixelData, outputImageData, width * height * pixelSize);

    // write the output bmp file
    if(!inputBitmap.write(outputImageName.c_str()))
    {
        std::cout << "Failed to write output image!" << std::endl;

        std::cout << OUTPUT_DIR << std::endl;
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}
///*******************************************************************

int
LocalEntropy::genBinaryImage()
{
    bifData binaryData;
    binaryData.kernelName = std::string("LocalEntropy_Kernels.cl");
    binaryData.flagsStr = std::string("");
    if(sampleArgs->isComplierFlagsSpecified())
    {
        binaryData.flagsFileName = std::string(sampleArgs->flags.c_str());
    }

    binaryData.binaryName = std::string(sampleArgs->dumpBinary.c_str());
    int status = generateBinaryImage(binaryData);
    return status;
}

std::vector<std::string> findBmpFiles(const std::string& folderPath){
    std::vector<std::string> bmpFiles;

    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bmp") {
                bmpFiles.push_back(entry.path().stem().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading directory: " << e.what() << std::endl;
    }

    return bmpFiles;
}


int
LocalEntropy::setupCL()
{
    cl_int err = CL_SUCCESS;
    cl_device_type dType;

    if(sampleArgs->deviceType.compare("cpu") == 0)
    {
        dType = CL_DEVICE_TYPE_CPU;
    }
    else //deviceType = "gpu"
    {
        dType = CL_DEVICE_TYPE_GPU;
        if(sampleArgs->isThereGPU() == false)
        {
            std::cout << "GPU not found. Falling back to CPU device" << std::endl;
            dType = CL_DEVICE_TYPE_CPU;
        }
    }

    /*
     * Have a look at the available platforms and pick either
     * the AMD one if available or a reasonable default.
     */
    err = cl::Platform::get(&platforms);
    CHECK_OPENCL_ERROR(err, "Platform::get() failed.");

    std::vector<cl::Platform>::iterator i;
    if(platforms.size() > 0)
    {
        if(sampleArgs->isPlatformEnabled())
        {
            i = platforms.begin() + sampleArgs->platformId;
        }
        else
        {
            for(i = platforms.begin(); i != platforms.end(); ++i)
            {
                if(!strcmp((*i).getInfo<CL_PLATFORM_VENDOR>().c_str(),
                           "Advanced Micro Devices, Inc."))
                {
                    break;
                }
            }
        }
    }

    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)(*i)(),
        0
    };

    context = cl::Context(dType, cps, NULL, NULL, &err);
    CHECK_OPENCL_ERROR(err, "Context::Context() failed.");

    devices = context.getInfo<CL_CONTEXT_DEVICES>(&err);
    CHECK_OPENCL_ERROR(err, "Context::getInfo() failed.");
    if (print_flag == 1){
        std::cout << "Platform :" << (*i).getInfo<CL_PLATFORM_VENDOR>().c_str() << "\n";
    }
    int deviceCount = (int)devices.size();
    int j = 0;
    if (print_flag == 1){
        for (std::vector<cl::Device>::iterator i = devices.begin(); i != devices.end();
        ++i, ++j)
        {
            std::cout << "Device " << j << " : ";
            std::string deviceName = (*i).getInfo<CL_DEVICE_NAME>();
            std::cout << deviceName.c_str() << "\n";
        }
        std::cout << "\n";
        }

    if (deviceCount == 0)
    {
        std::cout << "No device available\n";
        return SDK_FAILURE;
    }

    if(validateDeviceId(sampleArgs->deviceId, deviceCount))
    {
        std::cout << "validateDeviceId() failed" << std::endl;
        return SDK_FAILURE;
    }


    // Check for image support
    imageSupport = devices[sampleArgs->deviceId].getInfo<CL_DEVICE_IMAGE_SUPPORT>
                   (&err);
    CHECK_OPENCL_ERROR(err, "Device::getInfo() failed.");

    // If images are not supported then return
    if(!imageSupport)
    {
        OPENCL_EXPECTED_ERROR("Images are not supported on this device!");
    }

    commandQueue = cl::CommandQueue(context, devices[sampleArgs->deviceId], 0,
                                    &err);
    CHECK_OPENCL_ERROR(err, "CommandQueue::CommandQueue() failed.");
    /*
    * Create and initialize memory objects
    */
    inputImage2DGray = cl::Image2D(context,
                               CL_MEM_READ_ONLY,
                               cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                               width,
                               height,
                               0,
                               NULL,
                               &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (inputImage2DGray)");


    // Create memory objects for output Image
    outputImage2DGray = cl::Image2D(context,
                                CL_MEM_WRITE_ONLY,
                                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (outputImage2DGray)");

    /*
    * Create and initialize memory objects
    */
   inputImage2DEntropy = cl::Image2D(context,
                                CL_MEM_READ_WRITE,
                                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                width,
                                height,
                                0,
                                NULL,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (inputImage2DGray)");


    // Create memory objects for output Image
    outputImage2DEntropy = cl::Image2D(context,
                                CL_MEM_WRITE_ONLY,
                                cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8),
                                width,
                                height,
                                0,
                                0,
                                &err);
    CHECK_OPENCL_ERROR(err, "Image2D::Image2D() failed. (outputImage2DGray)");

    device.push_back(devices[sampleArgs->deviceId]);

    // create a CL program using the kernel source
    SDKFile kernelFile;
    std::string kernelPath = getPath();

    if(sampleArgs->isLoadBinaryEnabled())
    {
        kernelPath.append(sampleArgs->loadBinary.c_str());
        if(kernelFile.readBinaryFromFile(kernelPath.c_str()) != SDK_SUCCESS)
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }
        cl::Program::Binaries programBinary(1,std::make_pair(
                                                (const void*)kernelFile.source().data(),
                                                kernelFile.source().size()));

        program = cl::Program(context, device, programBinary, NULL, &err);
        CHECK_OPENCL_ERROR(err, "Program::Program(Binary) failed.");

    }
    else
    {
        kernelPath.append("LocalEntropy_Kernels.cl");
        if(!kernelFile.open(kernelPath.c_str()))
        {
            std::cout << "Failed to load kernel file : " << kernelPath << std::endl;
            return SDK_FAILURE;
        }

        // create program source
        cl::Program::Sources programSource(1,
                                           std::make_pair(kernelFile.source().data(),
                                                   kernelFile.source().size()));

        // Create program object
        program = cl::Program(context, programSource, &err);
        CHECK_OPENCL_ERROR(err, "Program::Program() failed.");

    }

    std::string flagsStr = std::string("");

    // Get additional options
    if(sampleArgs->isComplierFlagsSpecified())
    {
        SDKFile flagsFile;
        std::string flagsPath = getPath();
        flagsPath.append(sampleArgs->flags.c_str());
        if(!flagsFile.open(flagsPath.c_str()))
        {
            std::cout << "Failed to load flags file: " << flagsPath << std::endl;
            return SDK_FAILURE;
        }
        flagsFile.replaceNewlineWithSpaces();
        const char * flags = flagsFile.source().c_str();
        flagsStr.append(flags);
    }

    if(flagsStr.size() != 0)
    {
        std::cout << "Build Options are : " << flagsStr.c_str() << std::endl;
    }

    err = program.build(device, flagsStr.c_str());

    if(err != CL_SUCCESS)
    {
        if(err == CL_BUILD_PROGRAM_FAILURE)
        {
            std::string str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[sampleArgs->deviceId]);

            std::cout << " \n\t\t\tBUILD LOG\n";
            std::cout << " ************************************************\n";
            std::cout << str << std::endl;
            std::cout << " ************************************************\n";
        }
    }
    CHECK_OPENCL_ERROR(err, "Program::build() failed.");

    // Create kernel Histgram kernel
    grayscaleKernel = cl::Kernel(program, "toGrayscale", &err);
    CHECK_OPENCL_ERROR(err, "Failed to create grayscale kernel");

    entropyKernel = cl::Kernel(program, "entropy", &err);
    CHECK_OPENCL_ERROR(err, "Failed to create entropy kernel");
    

    // Check group size against group size returned by kernel
    kernelWorkGroupSize = grayscaleKernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>
                          (devices[sampleArgs->deviceId], &err);
    CHECK_OPENCL_ERROR(err, "Kernel::getWorkGroupInfo()  failed.");

    if((blockSizeX * blockSizeY) > kernelWorkGroupSize)
    {
        if(!sampleArgs->quiet)
        {
            std::cout << "Out of Resources!" << std::endl;
            std::cout << "Group Size specified : "
                      << blockSizeX * blockSizeY << std::endl;
            std::cout << "Max Group Size supported on the kernel : "
                      << kernelWorkGroupSize << std::endl;
            std::cout << "Falling back to " << kernelWorkGroupSize << std::endl;
        }

        if(blockSizeX > kernelWorkGroupSize)
        {
            blockSizeX = kernelWorkGroupSize;
            blockSizeY = 1;
        }
    }



    return SDK_SUCCESS;
}

int
LocalEntropy::runCLKernels()
{
    cl_int status;

    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;

    cl::Event writeEvt;
    status = commandQueue.enqueueWriteImage(
                inputImage2DGray,
                 CL_TRUE,
                 origin,
                 region,
                 0,
                 0,
                 inputImageData,
                 NULL,
                 &writeEvt);
    CHECK_OPENCL_ERROR(status,
                       "CommandQueue::enqueueWriteImage failed. (inputImage2DGray)");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    cl_int eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = writeEvt.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");

    }

    // Set appropriate arguments to the grayscale kernel
    // input buffer grayscale image
    status = grayscaleKernel.setArg(0, inputImage2DGray);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    // outBuffer imager
    status = grayscaleKernel.setArg(1, inputImage2DEntropy);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");


    /*
    * Enqueue a kernel run call.
    */
    cl::NDRange globalThreads(width, height);
    cl::NDRange localThreads(blockSizeX, blockSizeY);

    cl::Event histEvt;
    status = commandQueue.enqueueNDRangeKernel(
                grayscaleKernel,
                 cl::NullRange,
                 globalThreads,
                 localThreads,
                 0,
                 &histEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");


    // Set appropriate arguments to the kernel
    // input buffer image
    status = entropyKernel.setArg(0, inputImage2DEntropy);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (inputImageBuffer)");

    // outBuffer imager
    status = entropyKernel.setArg(1, outputImage2DEntropy);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (outputImageBuffer)");
    
    // structure element buffer
    cl::Buffer structElemBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        structElem.size() * sizeof(cl_uchar),
        structElem.data()
    );

    status = entropyKernel.setArg(2, structElemBuffer);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (structElemBuffer)");

    status = entropyKernel.setArg(3, mStruct);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (mStruct)");

    status = entropyKernel.setArg(4, nStruct);
    CHECK_OPENCL_ERROR(status, "Kernel::setArg() failed. (nStruct)");

    cl::Event entropyEvent;
    status = commandQueue.enqueueNDRangeKernel(
                 entropyKernel,
                 cl::NullRange,
                 globalThreads,
                 localThreads,
                 0,
                 &entropyEvent);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueNDRangeKernel() failed.");


    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = entropyEvent.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");
    }

    // Enqueue Read Image
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    region[0] = width;
    region[1] = height;
    region[2] = 1;

    // Enqueue readBuffer
    cl::Event readEvt;
    status = commandQueue.enqueueReadImage(
                outputImage2DEntropy,
                 CL_FALSE,
                 origin,
                 region,
                 0,
                 0,
                 outputImageData,
                 NULL,
                 &readEvt);
    CHECK_OPENCL_ERROR(status, "CommandQueue::enqueueReadImage failed.");

    status = commandQueue.flush();
    CHECK_OPENCL_ERROR(status, "cl::CommandQueue.flush failed.");

    eventStatus = CL_QUEUED;
    while(eventStatus != CL_COMPLETE)
    {
        status = readEvt.getInfo<cl_int>(
                     CL_EVENT_COMMAND_EXECUTION_STATUS,
                     &eventStatus);
        CHECK_OPENCL_ERROR(status,
                           "cl:Event.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS) failed.");

    }

    return SDK_SUCCESS;
}



int
LocalEntropy::initialize()
{
    // Call base class Initialize to get default configuration
    CHECK_ERROR(sampleArgs->initialize(), SDK_SUCCESS,
                "OpenCL resource initialization failed");

    // --iterations, -i
    Option* iteration_option = new Option;
    if(!iteration_option)
    {
        error("Memory Allocation error.\n");
        return SDK_FAILURE;
    }
    iteration_option->_sVersion = "i";
    iteration_option->_lVersion = "iterations";
    iteration_option->_description = "Number of iterations to execute kernel";
    iteration_option->_type = CA_ARG_INT;
    iteration_option->_value = &iterations;
    sampleArgs->AddOption(iteration_option);
    delete iteration_option;

    // --input
    Option* input_option = new Option;
    if(!input_option)
    {
        error("Memory Allocation error.\n");
        return SDK_FAILURE;
    }
    input_option->_sVersion = "g";
    input_option->_lVersion = "input";
    input_option->_description = "Input image name in format <image_name> without file extension. Has to be .bmp type!\nDefault input name: " + std::string(INPUT_IMAGE) + ".";
    input_option->_type = CA_ARG_STRING;
    input_option->_value = &inputName;
    sampleArgs->AddOption(input_option);
    delete input_option;

    // image or folder
    Option* image_option = new Option;
    if(!image_option)
    {
        error("Memory Allocation error.\n");
        return SDK_FAILURE;
    }
    image_option->_sVersion = "f";
    image_option->_lVersion = "folder";
    image_option->_description = "Input image folder name. If not set, input image is expected to be in the same folder as the executable.\nDefault input folder: " + std::string(INPUT_DIR) + ".";
    image_option->_type = CA_ARG_STRING;
    image_option->_value = &input_type;
    sampleArgs->AddOption(image_option);
    delete image_option;

    // --struct_elem, -s
    Option* struct_option = new Option;
    if(!struct_option)
    {
        error("Memory Allocation error.\n");
        return SDK_FAILURE;
    }
    struct_option->_sVersion = "s";
    struct_option->_lVersion = "struct_elem";
    struct_option->_description = "Structure element shape to perform filtration.\nAvailable shapes: \"circle\" (or \"c\"); \"elipse\" (or \"e\"); \"square\" (or \"sq\"); \"rectangle\" (or \"rec\")\nDefault structure element: " + std::string(DEF_STRUCT) + ".";
    struct_option->_type = CA_ARG_STRING; 
    struct_option->_value = &structShape;
    sampleArgs->AddOption(struct_option);
    delete struct_option;

    // --struct_m, -m
    Option* m_option = new Option;
    if(!m_option)
    {   
        error("Memory Allocation error.\n");
        return SDK_FAILURE;
    }
    m_option->_sVersion = "m";
    m_option->_lVersion = "struct_m";
    m_option->_description = "Height (number of rows) for structure element. Default: " + std::to_string(DEF_M) + ".";
    m_option->_type = CA_ARG_INT;
    m_option->_value = &mStruct;
    sampleArgs->AddOption(m_option);
    delete m_option;

    // --struct_n, -n
    Option* n_option = new Option;
    if(!n_option)
    {
        error("Memory Allocation error.\n");
        return SDK_FAILURE;
    }
    n_option->_sVersion = "n";
    n_option->_lVersion = "struct_n";
    n_option->_description = "Width (number of columns) for structure element. Default: " + std::to_string(DEF_N) + ".";
    n_option->_type = CA_ARG_INT;
    n_option->_value = &nStruct;
    sampleArgs->AddOption(n_option);
    delete n_option;


    return SDK_SUCCESS;
}

/**
* Generate structure element - m x n ellipse as 1D vector
*/
std::vector<cl_uchar> generateEllipse(int m, int n) {
    std::vector<cl_uchar> mask(m * n, 0);

    float a = (n-1) / 2.0f;              
    float b = (m-1) / 2.0f;           
    float x0 = a;
    float y0 = b;

    for (int y = 0; y < m; ++y) {
        for (int x = 0; x < n; ++x) {
            float dx = (x - x0) / a;
            float dy = (y - y0) / b;
            if (dx * dx + dy * dy <= 1.0f)
                mask[y * n + x] = 1;
        }
    }

    return mask;
}

/**
* Generate structure element - m x n rectangle as 1D vector
*/
std::vector<cl_uchar> generateRectangle(int m, int n) {
    return std::vector<cl_uchar>(m * n, 1);
}

/**
* Function to flatten 2D mask to 1D vector - unused, but can be useful to process manually created structure elements
*/
std::vector<cl_uchar> flatten(const std::vector<std::vector<cl_uchar>>& mask) {
    std::vector<cl_uchar> flatMask;
    for (const auto& row : mask) {
        flatMask.insert(flatMask.end(), row.begin(), row.end());
    }
    return flatMask;
}

int LocalEntropy::readInputImage_wrapper(){
    // Allocate host memory and read input image

    // Allocate host memory and read input image
    std::string filePath = getPath() + INPUT_DIR + inputName + ".bmp";
    if(readInputImage(filePath) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

}

int
LocalEntropy::setup()
{   
    //Short names of folder settings
    if (input_type == "d"){
        input_type = "directory";
    }
    else if (input_type == "i"){
        input_type = "image";
    }
    else{
        std::cout << "Unknown input type [available image/directory]. Defaulting to image." << std::endl;
        input_type = "image";
    }
    if (input_type == "directory" && Alt_ImageName != "default"){
        inputName = Alt_ImageName;
        is_dir = 1;
    }
    
    // Allocate host memory and read input image
    std::string filePath = getPath() + INPUT_DIR + inputName + ".bmp";
    if(readInputImage(filePath) != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    // Creating structure element
    // Setting m and n equal if one is not set
    if (mStruct <= 0 || nStruct <= 0) {
        if (mStruct >= nStruct){
            nStruct = mStruct;
        } else {
            mStruct = nStruct;
        }
    }

     



    // Short names of structure elements
    if (structShape == "sq")
        structShape = "square";
    else if (structShape == "rec")
        structShape = "rectangle";
    else if (structShape == "c")
        structShape = "circle";
    else if (structShape == "e")
        structShape = "ellipse";

    // Setting m and n equal if circle or square
    if (structShape == "square" || structShape == "circle") {
        if (mStruct >= nStruct){
            nStruct = mStruct;
        } else {
            mStruct = nStruct;
        }
    }

    // m or n < 0
    if (mStruct < 0 || nStruct < 0) {
        std::cout << "Failed to create structure element. m and n must be positive." << std::endl;
        return SDK_FAILURE;
    }

    // Setting default values if not set by user
    if (mStruct == 0) {
        // std::cout << "setting default value of m to " << std::to_string(DEF_M) << "." << std::endl;
        mStruct = DEF_M;
    }
    if (nStruct == 0) {
        // std::cout << "setting default value of n to " << std::to_string(DEF_N) << "." << std::endl;
        nStruct = DEF_N;
    }

    // create structure element
    if (structShape == "rectangle" || structShape == "square") {
        structElem = generateEllipse(mStruct, nStruct);
    } else if (structShape == "circle" || structShape == "ellipse") {
        structElem = generateRectangle(mStruct, nStruct);
    } else {
        std::cout << "Failed to create structure element. Unknown structure element name." << std::endl;
        return SDK_FAILURE;
    }

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    int status = setupCL();
    if (status != SDK_SUCCESS)
    {
        return status;
    }

    sampleTimer->stopTimer(timer);
    // Compute setup time
    setupTime = (double)(sampleTimer->readTimer(timer));

    return SDK_SUCCESS;

}

int
LocalEntropy::run()
{
    
    //if (inputFolder == "folder"){
    //    inputFolder = getPath() + INPUT_DIR;
    //} else {
    //   inputFolder = getPath() + inputFolder;
    //}

    if(!byteRWSupport)
    {
        return SDK_SUCCESS;
    }

    // Warm up
    for(int i = 0; i < 2 && iterations != 1; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

    }

    // Info for user
    std::cout << "Generating local entropy output image for input " + inputName + ".bmp with " + std::to_string(mStruct) + "x" + std::to_string(nStruct) + " " + structShape + " structure element."  << std::endl;

    std::cout << "Executing kernel for " << iterations
              << " iterations" <<std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    // create and initialize timers
    int timer = sampleTimer->createTimer();
    sampleTimer->resetTimer(timer);
    sampleTimer->startTimer(timer);

    for(int i = 0; i < iterations; i++)
    {
        // Set kernel arguments and run kernel
        if(runCLKernels() != SDK_SUCCESS)
        {
            return SDK_FAILURE;
        }

    }

    sampleTimer->stopTimer(timer);
    // Compute kernel time
    kernelTime = (double)(sampleTimer->readTimer(timer)) / iterations;

    // write the output image to bitmap file

    if(writeOutputImage(OUTPUT_DIR + ((INPUT_IMAGE == inputName) ? OUTPUT_IMAGE : (inputName + "_Output")) + "_" + structShape + std::to_string(mStruct) + "x" + std::to_string(nStruct) + ".bmp") != SDK_SUCCESS)
    {
        return SDK_FAILURE;
    }

    return SDK_SUCCESS;
}

int
LocalEntropy::cleanup()
{

    // release program resources (input memory etc.)
    FREE(inputImageData);
    FREE(outputImageData);
    FREE(verificationOutput);


    return SDK_SUCCESS;
}




///---------------------- NIE USUWAC -----------------------
void
LocalEntropy::printStats()
{
    if(sampleArgs->timing)
    {
        std::string strArray[4] =
        {
            "Width",
            "Height",
            "Time(sec)",
            "[Transfer+Kernel]Time(sec)"
        };
        std::string stats[4];

        sampleTimer->totalTime = setupTime + kernelTime;

        stats[0] = toString(width, std::dec);
        stats[1] = toString(height, std::dec);
        stats[2] = toString(sampleTimer->totalTime, std::dec);
        stats[3] = toString(kernelTime, std::dec);

        printStatistics(strArray, stats, 4);
    }
    sampleTimer->totalTime = setupTime + kernelTime;
    totalKernelTime_inSeconds = float(kernelTime);
    totalTime_inSeconds = float(sampleTimer->totalTime);
}
//-------------------------------------------------------------


int
main(int argc, char * argv[])
{

    auto start = std::chrono::high_resolution_clock::now();

    std::string folder = getPath() + INPUT_DIR;

    std::string first_image = "default_initial";

    auto files = findBmpFiles(folder);
    if (files.empty()){
        std::cout << "No .bmp files found in the folder: " << folder << std::endl;
        return SDK_FAILURE;
    }
    else{
        first_image = files[0];
        //std::cout << "First image: " << first_image << std::endl;
        std::cout << "Input images found in input folder:" << std::endl;
        for (const auto& file : files) {
            std::cout << file << std::endl;
        }
        std::cout << "===========================" << std::endl;
        int dir_flag = 1;
        int counter = 0;
        float totalTime_inSeconds = 0;
        float totalKernelTime_inSeconds = 0;
        for (const auto& file: files){
            // Main loop
            int print_flag = 1;
            if(counter > 0){
                print_flag = 0;
            }
            counter = counter + 1;
            LocalEntropy clLocalEntropy(file, print_flag);
            if(clLocalEntropy.initialize() != SDK_SUCCESS)
            {
                return SDK_FAILURE;
            }
            if(clLocalEntropy.sampleArgs->parseCommandLine(argc, argv))
            {
                return SDK_FAILURE;
            }
        
            if(clLocalEntropy.sampleArgs->isDumpBinaryEnabled())
            {
                return clLocalEntropy.genBinaryImage();
            }
            else
            {   
        
                // Setup
                int status = clLocalEntropy.setup();
                if(status != SDK_SUCCESS)
                {
                    return status;
                }
        
                // Run
                if(clLocalEntropy.run() != SDK_SUCCESS)
                {
                    return SDK_FAILURE;
                }
        
                //std::cout << "ended run" << std::endl;
        
                //std::cout << clLocalEntropy.getAltImgName() << std::endl;
        
                // Cleanup
                if(clLocalEntropy.cleanup() != SDK_SUCCESS)
                {
                    return SDK_FAILURE;
                }
        
                clLocalEntropy.printStats();



                dir_flag = clLocalEntropy.get_is_dir();

                totalTime_inSeconds = totalTime_inSeconds + clLocalEntropy.getTotalTime();
                totalKernelTime_inSeconds = totalKernelTime_inSeconds + clLocalEntropy.getTotalKernelTime();
                if (dir_flag == 0){
                    break;
                }
            }
        
        }
        if (dir_flag == 1){
            std::cout << "All images processed." << std::endl;

            auto end = std::chrono::high_resolution_clock::now();

            //calculate total time
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Total execution time [entire main]: " << duration_ms.count() << " ms" << std::endl;

            std::cout << "Total execution time [kernel]: " << std::setprecision(6) << totalKernelTime_inSeconds << " s" << std::endl;
            std::cout << "Total execution time [setup+kernel]: " << std::setprecision(6) << totalTime_inSeconds << " s" << std::endl;

        }
    }


    return SDK_SUCCESS;

}



