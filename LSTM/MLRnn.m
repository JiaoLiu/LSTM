//
//  MLRnn.m
//  LSTM
//
//  Created by Jiao Liu on 11/9/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLRnn.h"

@implementation MLRnn

#pragma mark - Inner Method

+ (double)truncated_normal:(double)mean dev:(double)stddev
{
    double outP = 0.0;
    do {
        static int hasSpare = 0;
        static double spare;
        if (hasSpare) {
            hasSpare = 0;
            outP = mean + stddev * spare;
            continue;
        }
        
        hasSpare = 1;
        static double u,v,s;
        do {
            u = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
            v = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
            s = u * u + v * v;
        } while ((s >= 1.0) || (s == 0.0));
        s = sqrt(-2.0 * log(s) / s);
        spare = v * s;
        outP = mean + stddev * u * s;
    } while (fabsl(outP) > 2*stddev);
    return outP;
}

+ (double *)fillVector:(double)num size:(int)size
{
    double *outP = malloc(sizeof(double) * size);
    vDSP_vfillD(&num, outP, 1, size);
    return outP;
    
}

+ (double *)weight_init:(int)size
{
    double *outP = malloc(sizeof(double) * size);
    for (int i = 0; i < size; i++) {
        outP[i] = [MLRnn truncated_normal:0 dev:0.1];
    }
    return outP;
}

+ (double *)bias_init:(int)size
{
    return [MLRnn fillVector:0.1f size:size];
}

+ (double *)tanh:(double *)input size:(int)size
{
    for (int i = 0; i < size; i++) {
        double num = input[i];
        if (num > 20) {
            input[i] = 1;
        }
        else if (num < -20)
        {
            input[i] = -1;
        }
        else
        {
            input[i] = (exp(num) - exp(num)) / (exp(num) + exp(num));
        }
    }
    return input;
}

#pragma mark - Init

- (id)initWithNodeNum:(int)num layerSize:(int)size dataDim:(int)dim
{
    self = [super init];
    if (self) {
        _nodeNum = num;
        _layerSize = size;
        _dataDim = dim;
        [self setupNet];
    }
    return self;
}

- (id)init
{
    self = [super init];
    if (self) {
        [self setupNet];
    }
    return self;
}

- (void)setupNet
{
    _inWeight = [MLRnn weight_init:_nodeNum * _dataDim];
    _outWeight = [MLRnn weight_init:_nodeNum * _dataDim];
    _flowWeight = [MLRnn weight_init:_nodeNum * _nodeNum];
    _outBias = calloc(_dataDim, sizeof(double));
    _flowBias = calloc(_nodeNum, sizeof(double));
    _output = calloc(_layerSize * _dataDim, sizeof(double));
    _state = calloc(_layerSize * _nodeNum, sizeof(double));
}

#pragma mark - Main Method

- (double *)forwardPropagation:(double *)input
{
    _input = input;
    // clean data
    double zero = 0;
    vDSP_vfillD(&zero, _output, 1, _layerSize * _dataDim);
    vDSP_vfillD(&zero, _state, 1, _layerSize * _nodeNum);
    
    for (int i = 0; i < _layerSize; i++) {
        double *temp1 = calloc(_nodeNum, sizeof(double));
        double *temp2 = calloc(_nodeNum, sizeof(double));
        if (i == 0) {
            vDSP_mmulD(_inWeight, 1, (input + i * _dataDim), 1, temp1, 1, _nodeNum, 1, _dataDim);
            vDSP_vaddD(temp1, 1,_flowBias, 1, temp1, 1, _nodeNum);
        }
        else
        {
            vDSP_mmulD(_inWeight, 1, (input + i * _dataDim), 1, temp1, 1, _nodeNum, 1, _dataDim);
            vDSP_mmulD(_flowWeight, 1, (_state + (i-1) * _nodeNum), 1, temp2, 1, _nodeNum, 1, _nodeNum);
            vDSP_vaddD(temp1, 1, temp2, 1, temp1, 1, _nodeNum);
            vDSP_vaddD(temp1, 1,_flowBias, 1, temp1, 1, _nodeNum);
        }
        [MLRnn tanh:temp1 size:_nodeNum];
        vDSP_vaddD((_state + i * _nodeNum), 1, temp1, 1, (_state + i * _nodeNum), 1, _nodeNum);
        vDSP_mmulD(_outWeight, 1, temp1, 1, (_output + i * _dataDim), 1, _dataDim, 1, _nodeNum);
        vDSP_vaddD((_output + i * _dataDim), 1, _outBias, 1,  (_output + i * _dataDim), 1, _dataDim);
        
        free(temp1);
        free(temp2);
    }
    
    return _output;
}

- (void)backPropagation:(double *)loss
{
    double *flowLoss = calloc(_nodeNum, sizeof(double));
    for (int i = _layerSize - 1; i >= 0 ; i--) {
        vDSP_vaddD(_outBias, 1, (loss + i * _dataDim), 1, _outBias, 1, _dataDim);
        double *transWeight = calloc(_nodeNum * _dataDim, sizeof(double));
        vDSP_mtransD(_outWeight, 1, transWeight, 1, _nodeNum, _dataDim);
        double *tanhLoss = calloc(_nodeNum, sizeof(double));
        vDSP_mmulD(transWeight, 1, (loss + i * _dataDim), 1, tanhLoss, 1, _nodeNum, 1, _dataDim);
        double *outWeightLoss = calloc(_nodeNum * _dataDim, sizeof(double));
        vDSP_mmulD((loss + i * _dataDim), 1, (_state + i * _nodeNum), 1, outWeightLoss, 1, _dataDim, _nodeNum, 1);
        vDSP_vaddD(_outWeight, 1, outWeightLoss, 1, _outWeight, 1, _nodeNum * _dataDim);
        
        double *tanhIn = calloc(_nodeNum, sizeof(double));
        vDSP_vsqD((_state + i * _nodeNum), 1, tanhIn, 1, _nodeNum);
        double *one = [MLRnn fillVector:1 size:_nodeNum];
        vDSP_vsubD(tanhIn, 1, one, 1, tanhIn, 1, _nodeNum);
        if (i != _layerSize - 1) {
            vDSP_vaddD(tanhLoss, 1, flowLoss, 1, tanhLoss, 1, _nodeNum);
        }
        vDSP_vmulD(tanhLoss, 1, tanhIn, 1, tanhLoss, 1, _nodeNum);
        
        vDSP_vaddD(_flowBias, 1, tanhLoss, 1, _flowBias, 1, _nodeNum);
        if (i != 0) {
            double *transFlow = calloc(_nodeNum * _nodeNum, sizeof(double));
            vDSP_mtransD(_flowWeight, 1, transFlow, 1, _nodeNum, _nodeNum);
            vDSP_mmulD(transFlow, 1, tanhLoss, 1, flowLoss, 1, _nodeNum, 1, _nodeNum);
            free(transFlow);
            double *flowWeightLoss = calloc(_nodeNum * _nodeNum, sizeof(double));
            vDSP_mmulD(tanhLoss, 1, (_state + (i-1) * _nodeNum), 1, flowWeightLoss, 1, _nodeNum, _nodeNum, 1);
            vDSP_vaddD(_flowWeight, 1, flowWeightLoss, 1, _flowWeight, 1, _nodeNum * _nodeNum);
            free(flowWeightLoss);
        }

        double *inWeightLoss = calloc(_nodeNum * _dataDim, sizeof(double));
        vDSP_mmulD(tanhLoss, 1, (_input + i * _dataDim), 1, inWeightLoss, 1, _nodeNum, _dataDim, 1);
        vDSP_vaddD(_inWeight, 1, inWeightLoss, 1, _inWeight, 1, _nodeNum * _dataDim);
        
        
//        for (int j = 0; j < _nodeNum * _dataDim; j++) {
//            printf("%f ", _inWeight[i * _nodeNum * _dataDim + j]);
//        }
//        printf("\n");
        free(transWeight);
        free(tanhLoss);
        free(outWeightLoss);
        free(tanhIn);
        free(one);
        free(inWeightLoss);
    }
    free(flowLoss);
    free(loss);
}

@end
