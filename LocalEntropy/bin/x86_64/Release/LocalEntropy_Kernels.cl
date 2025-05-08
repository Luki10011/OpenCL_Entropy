__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

/**
* Convert image from RGB to grayscale
* @param image2d_t inputImage - grayscale input image
* @param image2d_t outGrayImage - grayscale output image
*/
__kernel void toGrayscale(__read_only image2d_t inputImage, __write_only image2d_t outGrayImage)
{

	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 pixel = (0);
	float4 rgb2gray_coef = (float4)(0.2989, 0.5870, 0.1140, 0.0);
	//float4 Gy = Gx;
	
	//if( coord.x >= 1 && coord.x < (get_global_size(0)-1) && coord.y >= 1 && coord.y < get_global_size(1) - 1)
	//{
		// 0.2989 * R + 0.5870 * G + 0.1140 * B
		pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
		pixel = dot(pixel, rgb2gray_coef);

		write_imageui(outGrayImage, coord, convert_uint4(pixel));
	//}

			
}


__kernel void entropy(__read_only image2d_t inputGrayImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	const int size = 5;  // Neigborhood size - when 9 it is 9x9 square
	const int struct_n = size * size;  // Num of pixels in structure element

	uint localHist[256] = {0};
	uint4 pixel = (uint4)(0);

	// Iterating over Neighborhood - Computing local histogram
	for(int row = coord.x - (int)(size / 2.0f); row <= coord.x + (int)(size / 2.0f); ++row) {
		for(int col = coord.y - (int)(size / 2.0f); col <= coord.y + (int)(size / 2.0f); ++col) {
			pixel = read_imageui(inputGrayImage, imageSampler, (int2)(row, col));
			localHist[(uint) pixel.x] += 1;
		}
	}
	
	// Computing Entropy
	double entropyVal = 0.0;
	for (int i = 0; i < 256; ++i) {
		if (localHist[i] > 0) {
			double valProbability = (double) localHist[i] / struct_n;
			entropyVal -= valProbability * log2(valProbability);
		}
	}

	// Normalization 
	float maxEntropy = log2((double) (struct_n < 256 ? struct_n : 256)); // max entropy value for given window size (it depends on elements in window - struct_n)
	entropyVal = entropyVal * (256.0 / maxEntropy);

	// Writing to output image
	uint4 result = (uint4){entropyVal, entropyVal, entropyVal, 0};
	write_imageui(outputImage, coord, result);
}