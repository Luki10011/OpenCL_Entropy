// __constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR;
__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_NONE |
                                    CLK_FILTER_NEAREST;

/**
* Convert image from RGB to grayscale
* @param image2d_t inputImage - RGB (or gray) input image
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

/**
* Compute image local entropy using given structure element mask
* @param image2d_t inputGrayImage - grayscale input image
* @param image2d_t outputImage - grayscale output image - local entropy
* @param uchar* structElem - structure element as openCL buffer (2D mask flattened to 1D array)
* @param int m - height (number of rows) of structure element
* @param int n - width (number of cols) of structure element
* @param char structLetter - first letter of structure element name - 's' for square, 'r' for rectangle, 'c' for circle or 'e' for ellipse
*/
__kernel void entropy(__read_only image2d_t inputGrayImage, __write_only image2d_t outputImage, __global const uchar* structElem, const int m, const int n, const char structLetter)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	// printf("Struct elem in kernel for x = %d; y = %d.\n", coord.x, coord.y);
	// for (int y = 0; y < m; ++y) {
	// 	for (int x = 0; x < n; ++x) {
	// 		printf("%d ", structElem[y * n + x]);
	// 	}
	// 	printf("\n");
	// }

	uint localHist[256] = {0};
	uint4 pixel = (uint4)(0);

	int struct_n = 0;
	int i = 0; int j = 0;
	// Iterating over Neighborhood - Computing local histogram
	for(int row = coord.x - (int)(m / 2.0f); row <= coord.x + (int)(m / 2.0f); ++row) {
		for(int col = coord.y - (int)(n / 2.0f); col <= coord.y + (int)(n / 2.0f); ++col) {
			if (structElem[i * n + j]) {
				pixel = read_imageui(inputGrayImage, imageSampler, (int2)(row, col));
				++localHist[pixel.x];
				++struct_n;
			}
			++j;
		}
		++i;
		j = 0;
	}

	// int struct_n = 0;
	// int i = 0; int j = 0;
	// float a = (n-1) / 2.0f;              
	// float b = (m-1) / 2.0f;           
	// float x0 = a;
	// float y0 = b;
	// // Iterating over Neighborhood - Computing local histogram
	// for(int row = coord.x - (int)(m / 2.0f); row <= coord.x + (int)(m / 2.0f); ++row) {
	// 	for(int col = coord.y - (int)(n / 2.0f); col <= coord.y + (int)(n / 2.0f); ++col) {
	// 		// Ellipse equation - circle or ellipse
	// 		if (structLetter == 'e' || structLetter == 'c') {
	// 			float dx = (j - x0) / a;
	// 			float dy = (i - y0) / b;
	// 			if (dx * dx + dy * dy > 1.0f) {
	// 				// Skip pixel if in ellipse exterior
	// 				++j;
	// 				continue;
	// 			}
	// 		} 
	// 		// Pixel is always in rectagle or square
	// 		pixel = read_imageui(inputGrayImage, imageSampler, (int2)(row, col));
	// 		++localHist[pixel.x];
	// 		++struct_n;

	// 		++j;
	// 	}
	// 	++i;
	// 	j = 0;
	// }
	
	// Computing Entropy
	float entropyVal = 0.0f;
	for (int i = 0; i < 256; ++i) {
		if (localHist[i] > 0) {
			float valProbability = (float)localHist[i] / (float)struct_n;
			entropyVal -= valProbability * log2(valProbability);
		}
	}

	// Normalization 
	float maxEntropy = log2((float)(struct_n));
    float normEntropy = (maxEntropy > 0.0f) ? (entropyVal * (255.0f / maxEntropy)) : 0.0f;
    normEntropy = fmin(normEntropy, 255.0f);

	// Writing to output image
	uint4 result = (uint4){normEntropy, normEntropy, normEntropy, 255};
	write_imageui(outputImage, coord, result);
}
