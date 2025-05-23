__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_NONE |
                                    CLK_FILTER_NEAREST;

/**
* @brief Konwersja obrazu z RGB do skali szarości
* @param inputImage - obraz wejściowy RGB (lub w skali szarości)
* @param outGrayImage - obraz wyjściowy w skali szarości
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
* @brief Oblicza lokalną entropię obrazu z użyciem podanej maski elementu strukturalnego
* @param inputGrayImage - obraz wejściowy w skali szarości
* @param outputImage - obraz wyjściowy w skali szarości - lokalna entropia
* @param structElem - element strukturalny jako bufor OpenCL (2D maska spłaszczona do tablicy 1D)
* @param m - wysokość (liczba wierszy) elementu strukturalnego
* @param n - szerokość (liczba kolumn) elementu strukturalnego
* @param structLetter - pierwsza litera nazwy elementu strukturalnego - 's' dla kwadratu, 'r' dla prostokąta, 'c' dla koła lub 'e' dla elipsy. 
* w obecnej wersji programu parametr nie jest używany - przydatny do zadania generowania elementu strukturalnego w czasie działania kernela.
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
