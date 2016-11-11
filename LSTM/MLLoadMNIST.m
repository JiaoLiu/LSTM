//
//  MLLoadMNIST.m
//  MNIST
//
//  Created by Jiao Liu on 9/23/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLLoadMNIST.h"

@implementation MLLoadMNIST

int reverseInt(int input)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=input&255;
    ch2=(input>>8)&255;
    ch3=(input>>16)&255;
    ch4=(input>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

- (double **)readImageData:(const char *)filePath
{
    FILE *file = fopen(filePath, "rb");
    double **output = NULL;
    if (file) {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        fread((char*)&magic_number, sizeof(magic_number), 1, file);
        magic_number= reverseInt(magic_number);
        fread((char*)&number_of_images, sizeof(number_of_images), 1, file);
        number_of_images= reverseInt(number_of_images);
        fread((char*)&n_rows, sizeof(n_rows), 1, file);
        n_rows= reverseInt(n_rows);
        fread((char*)&n_cols, sizeof(n_cols), 1, file);
        n_cols= reverseInt(n_cols);
        output = (double **)malloc(sizeof(double) * number_of_images);
        for(int i=0;i<number_of_images;++i)
        {
            output[i] = (double *)malloc(sizeof(double) * n_rows * n_cols);
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    fread((char*)&temp, sizeof(temp), 1, file);
                    output[i][(n_rows*r)+c]= (double)temp;
                }
            }
        }
    }
    fclose(file);
    return output;
}

- (int *)readLabelData:(const char *)filePath
{
    FILE *file = fopen(filePath, "rb");
    int *output = NULL;
    if (file) {
        int magic_number=0;
        int number_of_items=0;
        fread((char*)&magic_number, sizeof(magic_number), 1, file);
        magic_number= reverseInt(magic_number);
        fread((char*)&number_of_items, sizeof(number_of_items), 1, file);
        number_of_items= reverseInt(number_of_items);
        output = (int *)malloc(sizeof(int) * number_of_items);
        for(int i=0;i<number_of_items;++i)
        {
            unsigned char temp=0;
            fread((char*)&temp, sizeof(temp), 1, file);
            output[i]= (int)temp;
        }
    }
    fclose(file);
    return output;
}

@end
