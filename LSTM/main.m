//
//  main.m
//  LSTM
//
//  Created by Jiao Liu on 11/9/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "MLSoftMax.h"
#import "MLLoadMNIST.h"

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        MLLoadMNIST *loader = [[MLLoadMNIST alloc] init];
        double **trainImage = [loader readImageData:("/Users/Jiao/Desktop/SecurityKeeper/MNIST/train-images-idx3-ubyte")];
        int *trainLabel = [loader readLabelData:("/Users/Jiao/Desktop/SecurityKeeper/MNIST/train-labels-idx1-ubyte")];
        
        MLSoftMax *softMax = [[MLSoftMax alloc] initWithLoopNum:2000 dim:784 type:10 size:100 descentRate:0.01];
        softMax.image = trainImage;
        softMax.label = trainLabel;
        softMax.trainNum = 60000;
        [softMax train];
        
        printf("complete training!\n");

        double **testImage = [loader readImageData:("/Users/Jiao/Desktop/SecurityKeeper/MNIST/t10k-images-idx3-ubyte")];
        int *testLabel = [loader readLabelData:("/Users/Jiao/Desktop/SecurityKeeper/MNIST/t10k-labels-idx1-ubyte")];
        
        int correct = 0;
        for (int i = 0; i < 1e4; i++) {
            int pred = [softMax predict:testImage[i]];
            if (pred == testLabel[i]) {
                correct++;
            }
            printf("%d - %d \n",testLabel[i],pred);
        }
        printf("%f\n",correct / 10000.0);
    }
    return 0;
}
