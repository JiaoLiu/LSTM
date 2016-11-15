//
//  MLLstm.h
//  LSTM
//
//  Created by Jiao Liu on 11/12/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>

@interface MLLstm : NSObject
{
    @private
    double *_input;
    double *_hState;
    double *_rState;
    double *_zState;
    double *_hbState;
    double *_output;
    double *_backLoss;
    
    double *_rW;
    double *_rU;
    double *_rBias;
    double *_zW;
    double *_zU;
    double *_zBias;
    double *_hW;
    double *_hU;
    double *_hBias;
    double *_outW;
    double *_outBias;
}

@property (nonatomic, assign)int nodeNum; // num of node in each neuron
@property (nonatomic, assign)int layerSize; // num of neurons in each layer
@property (nonatomic, assign)int dataDim;

- (id)initWithNodeNum:(int)num layerSize:(int)size dataDim:(int)dim;
- (double *)forwardPropagation:(double *)input;
- (double *)backPropagation:(double *)loss;

@end
