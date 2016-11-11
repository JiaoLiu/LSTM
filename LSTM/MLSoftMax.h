//
//  MLSoftMax.h
//  MNIST
//
//  Created by Jiao Liu on 9/26/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import "MLRnn.h"

@interface MLSoftMax : NSObject
{
    @private
    int _iterNum;
    int _randSize;
    double _descentRate;
    double **_randomX;
    int *_randomY;
    double *_theta;
    double *_bias;
    MLRnn *_rnn;
}

@property (nonatomic, assign) double **image;
@property (nonatomic, assign) int *label;
@property (nonatomic, assign) int trainNum;
@property (nonatomic, assign) int kType;
@property (nonatomic, assign) int dim;

- (id)initWithLoopNum:(int)loopNum dim:(int)dim type:(int)type size:(int)size descentRate:(double)rate;
- (void)train;
- (int)predict:(double *)image;
- (void)saveTrainDataToDisk;
- (int)predict:(double *)image withOldTheta:(double *)theta andBias:(double *)bias;
- (void)updateModel:(double *)image label:(int)label;

@end
