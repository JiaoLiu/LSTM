//
//  MLLstm.m
//  LSTM
//
//  Created by Jiao Liu on 11/12/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLLstm.h"

@implementation MLLstm

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
        outP[i] = [MLLstm truncated_normal:0 dev:0.1];
    }
    return outP;
}

+ (double *)bias_init:(int)size
{
    return [MLLstm fillVector:0.1f size:size];
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
            input[i] = (exp(num) - exp(-num)) / (exp(num) + exp(-num));
        }
    }
    return input;
}

+ (double *)sigmoid:(double *)input size:(int)size
{
    for (int i = 0; i < size; i++) {
        double num = input[i];
        if (num > 20) {
            input[i] = 1;
        }
        else if (num < -20)
        {
            input[i] = 0;
        }
        else
        {
            input[i] = exp(num) / (exp(num) + 1);
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
    _hState = calloc(_layerSize * _nodeNum, sizeof(double));
    _rState = calloc(_layerSize * _nodeNum, sizeof(double));
    _zState = calloc(_layerSize * _nodeNum, sizeof(double));
    _hbState = calloc(_layerSize * _nodeNum, sizeof(double));
    _output = calloc(_layerSize * _dataDim, sizeof(double));
    _backLoss = calloc(_layerSize * _dataDim, sizeof(double));
    
    _rW = [MLLstm weight_init:_nodeNum * _dataDim];
    _rU = [MLLstm weight_init:_nodeNum * _nodeNum];
    _rBias = [MLLstm bias_init:_nodeNum];
    _zW = [MLLstm weight_init:_nodeNum * _dataDim];
    _zU = [MLLstm weight_init:_nodeNum * _nodeNum];
    _zBias = [MLLstm bias_init:_nodeNum];
    _hW = [MLLstm weight_init:_nodeNum * _dataDim];
    _hU = [MLLstm weight_init:_nodeNum * _nodeNum];
    _hBias = [MLLstm bias_init:_nodeNum];
    _outW = [MLLstm weight_init:_dataDim * _nodeNum];
    _outBias = [MLLstm bias_init:_dataDim];
}

- (double *)forwardPropagation:(double *)input
{
    _input = input;
    // clean data
    double zero = 0;
    vDSP_vfillD(&zero, _output, 1, _layerSize * _dataDim);
    vDSP_vfillD(&zero, _hState, 1, _layerSize * _nodeNum);
    vDSP_vfillD(&zero, _rState, 1, _layerSize * _nodeNum);
    vDSP_vfillD(&zero, _zState, 1, _layerSize * _nodeNum);
    vDSP_vfillD(&zero, _hbState, 1, _layerSize * _nodeNum);
    vDSP_vfillD(&zero, _backLoss, 1, _layerSize * _dataDim);
    
    double *temp1 = calloc(_nodeNum, sizeof(double));
    double *temp2 = calloc(_nodeNum, sizeof(double));
    double *temp3 = calloc(_nodeNum, sizeof(double));
    double *one = [MLLstm fillVector:1 size:_nodeNum];
    for (int i = 0; i < _layerSize; i++) {
        //rj =σ  [Wr*(xt)]j +  Ur*h⟨t−1⟩ + rBias]
        if (i == 0) {
            vDSP_mmulD(_rW, 1, (_input + i * _dataDim), 1, temp1, 1, _nodeNum, 1, _dataDim);
            vDSP_vaddD(temp1, 1, _rBias, 1, temp1, 1, _nodeNum);
        }
        else
        {
            vDSP_mmulD(_rW, 1, (_input + i * _dataDim), 1, temp1, 1, _nodeNum, 1, _dataDim);
            vDSP_mmulD(_rU, 1, (_hState + (i-1) * _nodeNum), 1, temp2, 1, _nodeNum, 1, _nodeNum);
            vDSP_vaddD(temp1, 1, temp2, 1, temp1, 1, _nodeNum);
            vDSP_vaddD(temp1, 1, _rBias, 1, temp1, 1, _nodeNum);
        }
        [MLLstm sigmoid:temp1 size:_nodeNum];
        vDSP_vaddD((_rState + i * _nodeNum), 1, temp1, 1, (_rState + i * _nodeNum), 1, _nodeNum);
        
        //zj =σ  [Wz*(xt)]j +  Uz*h⟨t−1⟩ + zBias]
        if (i == 0) {
            vDSP_mmulD(_zW, 1, (_input + i * _dataDim), 1, temp1, 1, _nodeNum, 1, _dataDim);
            vDSP_vaddD(temp1, 1, _zBias, 1, temp1, 1, _nodeNum);
        }
        else
        {
            vDSP_mmulD(_zW, 1, (_input + i * _dataDim), 1, temp1, 1, _nodeNum, 1, _dataDim);
            vDSP_mmulD(_zU, 1, (_hState + (i-1) * _nodeNum), 1, temp2, 1, _nodeNum, 1, _nodeNum);
            vDSP_vaddD(temp1, 1, temp2, 1, temp1, 1, _nodeNum);
            vDSP_vaddD(temp1, 1, _zBias, 1, temp1, 1, _nodeNum);
        }
        [MLLstm sigmoid:temp1 size:_nodeNum];
        vDSP_vaddD((_zState + i * _nodeNum), 1, temp1, 1, (_zState + i * _nodeNum), 1, _nodeNum);
        
        //h ̃⟨t⟩ = tanh  {[W*(xt)] +  U * [r ⊙ h⟨t−1⟩] + hBias}
        if (i == 0) {
            vDSP_mmulD(_hW, 1, (_input + i * _dataDim), 1, temp1, 1, _nodeNum, 1, _dataDim);
            vDSP_vaddD(temp1, 1, _hBias, 1, temp1, 1, _nodeNum);
        }
        else
        {
            vDSP_mmulD(_hW, 1, (_input + i * _dataDim), 1, temp1, 1, _nodeNum, 1, _dataDim);
            vDSP_vmulD((_rState + i * _nodeNum), 1, (_hState + (i-1) * _nodeNum), 1, temp2, 1, _nodeNum);
            vDSP_mmulD(_hU, 1, temp2, 1, temp3, 1, _nodeNum, 1, _nodeNum);
            vDSP_vaddD(temp1, 1, temp3, 1, temp1, 1, _nodeNum);
            vDSP_vaddD(temp1, 1, _hBias, 1, temp1, 1, _nodeNum);
        }
        [MLLstm tanh:temp1 size:_nodeNum];
        vDSP_vaddD((_hbState + i * _nodeNum), 1, temp1, 1, (_hbState + i * _nodeNum), 1, _nodeNum);
        
        //h⟨t⟩ = zj⊙ h⟨t−1⟩ + (1 − zj)⊙ h ̃⟨t⟩
        if (i == 0) {
            vDSP_vsubD((_zState + i * _nodeNum), 1, one, 1, temp1, 1, _nodeNum);
            vDSP_vmulD((_hbState + i * _nodeNum), 1, temp1, 1, temp1, 1, _nodeNum);
        }
        else
        {
            vDSP_vsubD((_zState + i * _nodeNum), 1, one, 1, temp1, 1, _nodeNum);
            vDSP_vmulD((_hbState + i * _nodeNum), 1, temp1, 1, temp1, 1, _nodeNum);
            vDSP_vmulD((_zState + i * _nodeNum), 1, (_hState + (i-1) * _nodeNum), 1, temp2, 1, _nodeNum);
            vDSP_vaddD(temp1, 1, temp2, 1, temp1, 1, _nodeNum);
        }
        vDSP_vaddD((_hState + i * _nodeNum), 1, temp1, 1, (_hState + i * _nodeNum), 1, _nodeNum);
        
        // output
        vDSP_mmulD(_outW, 1, (_hState + i * _nodeNum), 1, (_output + i * _dataDim), 1, _dataDim, 1, _nodeNum);
        vDSP_vaddD(_outBias, 1, (_output + i * _dataDim), 1, (_output + i * _dataDim), 1, _dataDim);
    }
    free(one);
    free(temp1);
    free(temp2);
    free(temp3);
    
    return _output;
}

- (double *)backPropagation:(double *)loss
{
    double *flowLoss = calloc(_nodeNum, sizeof(double));
    double *outTW = calloc(_nodeNum * _dataDim, sizeof(double));
    double *outLoss = calloc(_nodeNum, sizeof(double));
    double *outWLoss = calloc(_dataDim * _nodeNum, sizeof(double));
    double *temp1 = calloc(_nodeNum, sizeof(double));
    double *one = [MLLstm fillVector:1 size:_nodeNum];
    double *zLoss = calloc(_nodeNum, sizeof(double));
    double *hbLoss = calloc(_nodeNum, sizeof(double));
    double *inWLoss = calloc(_nodeNum * _dataDim, sizeof(double));
    double *rLoss = calloc(_nodeNum, sizeof(double));
    double *tU = calloc(_nodeNum * _nodeNum, sizeof(double));
    double *uLoss = calloc(_nodeNum * _nodeNum, sizeof(double));
    double *tW = calloc(_dataDim * _nodeNum, sizeof(double));
    double *temp2 = calloc(_dataDim, sizeof(double));
    for (int i = _layerSize - 1; i >= 0; i--) {
        // update output parameters
        vDSP_vaddD(_outBias, 1, (loss + i * _dataDim), 1, _outBias, 1, _dataDim);
        vDSP_mtransD(_outW, 1, outTW, 1, _nodeNum, _dataDim);
        vDSP_mmulD(outTW, 1, (loss + i * _dataDim), 1, outLoss, 1, _nodeNum, 1, _dataDim);
        vDSP_mmulD((loss + i * _dataDim), 1, (_hState + i * _nodeNum), 1, outWLoss, 1, _dataDim, _nodeNum, 1);
        vDSP_vaddD(_outW, 1, outWLoss, 1, _outW, 1, _dataDim * _nodeNum);
        
        // h(t) back loss
        if (i != _layerSize - 1) {
            vDSP_vaddD(outLoss, 1, flowLoss, 1, outLoss, 1, _nodeNum);
        }
        if (i > 0) {
            vDSP_vsubD((_hState + (i-1) * _nodeNum), 1, (_hbState + i * _nodeNum), 1, temp1, 1, _nodeNum);
            vDSP_vmulD(outLoss, 1, temp1, 1, zLoss, 1, _nodeNum);
            
            vDSP_vsubD((_zState + i * _nodeNum), 1, one, 1, temp1, 1, _nodeNum);
            vDSP_vmulD(outLoss, 1, temp1, 1, flowLoss, 1, _nodeNum);
        }
        else
        {
            vDSP_vmulD(outLoss, 1, (_hbState + i * _nodeNum), 1, zLoss, 1, _nodeNum);
        }
        // σ` = f(x)*(1-f(x))
        vDSP_vsubD((_zState + i * _nodeNum), 1, one, 1, temp1, 1, _nodeNum);
        vDSP_vmulD(temp1, 1, (_zState + i * _nodeNum), 1, temp1, 1, _nodeNum);
        vDSP_vmulD(temp1, 1, zLoss, 1, zLoss, 1, _nodeNum);
        
        vDSP_vmulD(outLoss, 1, (_zState + i * _nodeNum), 1, hbLoss, 1, _nodeNum);
        // tanh` =  1-f(x)**2
        vDSP_vsqD((_hbState + i * _nodeNum), 1, temp1, 1, _nodeNum);
        vDSP_vsubD(temp1, 1, one, 1, temp1, 1, _nodeNum);
        vDSP_vmulD(hbLoss, 1, temp1, 1, hbLoss, 1, _nodeNum);
        
        // update h`(t) parameters
        vDSP_vaddD(_hBias, 1, hbLoss, 1, _hBias, 1, _nodeNum);
        vDSP_mtransD(_hW, 1, tW, 1, _dataDim, _nodeNum);
        vDSP_mmulD(tW, 1, hbLoss, 1, temp2, 1, _dataDim, 1, _nodeNum);
        vDSP_vaddD((_backLoss + i * _dataDim), 1, temp2, 1, (_backLoss + i * _dataDim), 1, _dataDim);
        vDSP_mmulD(hbLoss, 1, (_input + i * _dataDim), 1, inWLoss, 1, _nodeNum, _dataDim, 1);
        vDSP_vaddD(_hW, 1, inWLoss, 1, _hW, 1, _nodeNum * _dataDim);

        if (i > 0) {
            vDSP_mtransD(_hU, 1, tU, 1, _nodeNum, _nodeNum);
            vDSP_mmulD(tU, 1, hbLoss, 1, rLoss, 1, _nodeNum, 1, _nodeNum);
            vDSP_vmulD(rLoss, 1, (_hState + (i-1) * _nodeNum), 1, rLoss, 1, _nodeNum);
            vDSP_vsubD((_rState + i * _nodeNum), 1, one, 1, temp1, 1, _nodeNum);
            vDSP_vmulD(temp1, 1, (_rState + i * _nodeNum), 1, temp1, 1, _nodeNum);
            vDSP_vmulD(temp1, 1, rLoss, 1, rLoss, 1, _nodeNum);
            
            vDSP_mmulD(tU, 1, hbLoss, 1, temp1, 1, _nodeNum, 1, _nodeNum);
            vDSP_vmulD(temp1, 1, (_rState + i * _nodeNum), 1, temp1, 1, _nodeNum);
            vDSP_vaddD(flowLoss, 1, temp1, 1, flowLoss, 1, _nodeNum);
            
            vDSP_vmulD((_rState + i * _nodeNum), 1, (_hState + (i-1) * _nodeNum), 1, temp1, 1, _nodeNum);
            vDSP_mmulD(hbLoss, 1, temp1, 1, uLoss, 1, _nodeNum, _nodeNum, 1);
            vDSP_vaddD(_hU, 1, uLoss, 1, _hU, 1, _nodeNum * _nodeNum);
        }
        
        // update z(t) parameters
        vDSP_vaddD(_zBias, 1, zLoss, 1, _zBias, 1, _nodeNum);
        vDSP_mtransD(_zW, 1, tW, 1, _dataDim, _nodeNum);
        vDSP_mmulD(tW, 1, zLoss, 1, temp2, 1, _dataDim, 1, _nodeNum);
        vDSP_vaddD((_backLoss + i * _dataDim), 1, temp2, 1, (_backLoss + i * _dataDim), 1, _dataDim);
        vDSP_mmulD(zLoss, 1, (_input + i * _dataDim), 1, inWLoss, 1, _nodeNum, _dataDim, 1);
        vDSP_vaddD(_zW, 1, inWLoss, 1, _zW, 1, _nodeNum * _dataDim);
        
        if (i > 0) {
            vDSP_mtransD(_zU, 1, tU, 1, _nodeNum, _nodeNum);
            vDSP_mmulD(tU, 1, zLoss, 1, temp1, 1, _nodeNum, 1, _nodeNum);
            vDSP_vaddD(flowLoss, 1, temp1, 1, flowLoss, 1, _nodeNum);
            
            vDSP_mmulD(zLoss, 1, (_hState + (i-1) * _nodeNum), 1, uLoss, 1, _nodeNum, _nodeNum, 1);
            vDSP_vaddD(_zU, 1, uLoss, 1, _zU, 1, _nodeNum * _nodeNum);
        }
        
        // update r(t) parameters
        if (i > 0) {
            vDSP_vaddD(_rBias, 1, rLoss, 1, _rBias, 1, _nodeNum);
            vDSP_mtransD(_rW, 1, tW, 1, _dataDim, _nodeNum);
            vDSP_mmulD(tW, 1,rLoss, 1, temp2, 1, _dataDim, 1, _nodeNum);
            vDSP_vaddD((_backLoss + i * _dataDim), 1, temp2, 1, (_backLoss + i * _dataDim), 1, _dataDim);
            vDSP_mmulD(rLoss, 1, (_input + i * _dataDim), 1, inWLoss, 1, _nodeNum, _dataDim, 1);
            vDSP_vaddD(_rW, 1, inWLoss, 1, _rW, 1, _nodeNum * _dataDim);
            
            vDSP_mtransD(_rU, 1, tU, 1, _nodeNum, _nodeNum);
            vDSP_mmulD(tU, 1, rLoss, 1, temp1, 1, _nodeNum, 1, _nodeNum);
            vDSP_vaddD(flowLoss, 1, temp1, 1, flowLoss, 1, _nodeNum);
            
            vDSP_mmulD(rLoss, 1, (_hState + (i-1) * _nodeNum), 1, uLoss, 1, _nodeNum, _nodeNum, 1);
            vDSP_vaddD(_rU, 1, uLoss, 1, _rU, 1, _nodeNum * _nodeNum);
        }
    }
    
    free(flowLoss);
    free(outTW);
    free(outLoss);
    free(outWLoss);
    free(temp1);
    free(one);
    free(zLoss);
    free(hbLoss);
    free(inWLoss);
    free(rLoss);
    free(tU);
    free(uLoss);
    free(tW);
    free(temp2);
    return _backLoss;
}

@end
