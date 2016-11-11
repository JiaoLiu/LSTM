//
//  MLRnn.h
//  LSTM
//
//  Created by Jiao Liu on 11/9/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

@interface MLRnn : NSObject
{
    @private
    double *_inWeight;
    double *_outWeight;
    double *_flowWeight;
    double *_outBias;
    double *_flowBias;
    double *_output;
    double *_state;
    double *_input;
}

@property (nonatomic, assign)int nodeNum; // num of node in each neuron
@property (nonatomic, assign)int layerSize; // num of neurons in each layer
@property (nonatomic, assign)int dataDim;

- (id)initWithNodeNum:(int)num layerSize:(int)size dataDim:(int)dim;
- (double *)forwardPropagation:(double *)input;
- (void)backPropagation:(double *)loss;

@end
