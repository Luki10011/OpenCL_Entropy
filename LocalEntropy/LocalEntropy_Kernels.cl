__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

__kernel void histogram(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 pixel = (float4)(0);
	//float4 Gy = Gx;
	
	//if( coord.x >= 1 && coord.x < (get_global_size(0)-1) && coord.y >= 1 && coord.y < get_global_size(1) - 1)
	//{

		pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));

		write_imageui(outputImage, coord, convert_uint4(pixel));
	//}

			
}


__kernel void entropy(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float4 pixel = (float4)(0);
	//float4 Gy = Gx;
	
	//if( coord.x >= 1 && coord.x < (get_global_size(0)-1) && coord.y >= 1 && coord.y < get_global_size(1) - 1)
	//{

		pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
		
		pixel = (float4)256.0f - pixel;

		write_imageui(outputImage, coord, convert_uint4(pixel));
	//}

			
}