//
//  MLSoftMax.m
//  MNIST
//
//  Created by Jiao Liu on 9/26/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import "MLSoftMax.h"

@implementation MLSoftMax

- (id)initWithLoopNum:(int)loopNum dim:(int)dim type:(int)type size:(int)size descentRate:(double)rate
{
    self = [super init];
    if (self) {
        _iterNum = loopNum == 0 ? 500 : loopNum;
        _dim = dim;
        _kType = type;
        _randSize = size == 0 ? 100 : size;
        _descentRate = rate == 0 ? 0.01 : rate;
//        _bias = [MLCnn bias_init:type];
//        _theta = [MLCnn weight_init:type * dim];
        _bias = malloc(sizeof(double) * type);
        _theta = malloc(sizeof(double) * type * dim);
        double fillNum = 0.0f;
        vDSP_vfillD(&fillNum, _bias, 1, type);
        vDSP_vfillD(&fillNum, _theta, 1, type * dim);
//        _rnn = [[MLRnn alloc] initWithNodeNum:50 layerSize:28 dataDim:28];
        _lstm = [[MLLstm alloc] initWithNodeNum:100 layerSize:28 dataDim:28];
    }
    return  self;
}

- (void)dealloc
{
    if (_bias != NULL) {
        free(_bias);
        _bias = NULL;
    }
    
    if (_theta != NULL) {
        free(_theta);
        _theta = NULL;
    }
    
    if (_randomX != NULL) {
        free(_randomX);
        _randomX = NULL;
    }
    
    if (_randomY != NULL) {
        free(_randomY);
        _randomY = NULL;
    }
}

#pragma mark - SoftMax Main

- (void)randomPick:(int)maxSize
{
    long rNum = random();
    for (int i = 0; i < _randSize; i++) {
        _randomX[i] = _image[(rNum+i) % maxSize];
//        for (int j = 0; j < _dim; j++) {
//            printf("%f ",_randomX[i][j]);
//        }
//        printf("\n\n");
        _randomY[i] = _label[(rNum+i) % maxSize];
    }
}

- (double *)MaxPro:(double *)index
{
    double maxNum = 0;
    vDSP_maxvD(index, 1, &maxNum, _kType);
    
    double sum = 0;
    for (int i = 0; i < _kType; i++) {
        index[i] -= maxNum;
        index[i] = expl(index[i]);
        sum += index[i];
//        printf("%f ",index[i]);
    }
//    printf("----\n");
    
    vDSP_vsdivD(index, 1, &sum, index, 1, _kType);
    return index;
}

- (void)updateModel:(double *)index currentPos:(int)pos
{
    for (int i = 0; i < _kType; i++) {
        double delta;
        if (i != _randomY[pos]) {
            delta = 0.0 - index[i];
        }
        else
        {
            delta = 1.0 - index[i];
        }
        
        _bias[i] += _descentRate * delta;
        double loss = _descentRate * delta / _randSize;
        double *decay = malloc(sizeof(double) * _dim);
        vDSP_vsmulD(_randomX[pos], 1, &loss, decay, 1, _dim);
        double *backLoss = malloc(sizeof(double) * _dim);
        vDSP_vsmulD((_theta + i * _dim), 1, &loss, backLoss, 1, _dim);
//        for (int k = 0; k < _dim; k++) {
//            printf("%f ", backLoss[k]);
//        }
//        printf("\n%d-------\n",pos);
        [_lstm backPropagation:backLoss];
//        [_rnn backPropagation:backLoss];
//        [_rnn backPropagation:[_lstm backPropagation:backLoss]];
        free(backLoss);
        
        vDSP_vaddD((_theta + i * _dim), 1, decay, 1, (_theta + i * _dim), 1, _dim);
        if (decay != NULL) {
            free(decay);
            decay = NULL;
        }
    }
}

- (void)train
{
    _randomX = malloc(sizeof(double) * _randSize);
    _randomY = malloc(sizeof(int) * _randSize);
    double *index = malloc(sizeof(double) * _kType);
    
    for (int i = 0; i < _iterNum; i++) {
        [self randomPick:_trainNum];
        for (int j = 0; j < _randSize; j++) {
            // calculate wx+b
//            _randomX[j] = [_rnn forwardPropagation:_randomX[j]];
            _randomX[j] = [_lstm forwardPropagation:_randomX[j]];
            vDSP_mmulD(_theta, 1, _randomX[j], 1, index, 1, _kType, 1, _dim);
            vDSP_vaddD(index, 1, _bias, 1, index, 1, _kType);
            // calulate exp(wx+b) / sum(exp(wx+b))
            index = [self MaxPro:index];
            [self updateModel:index currentPos:j];
//            for (int m = 0; m < _kType; m++) {
//                for (int n = 0; n < _dim; n++) {
//                    printf("%f ", _theta[m*_dim + n]);
//                }
//                printf("\n%d:=====\n",j);
//            }
        }
        if (i%100 == 0) {
            [self testModel];
        }
    }
    if (index != NULL) {
        free(index);
        index = NULL;
    }
}

/*
- (int)indicator:(int)label var:(int)x
{
    if (label == x) {
        return 1;
    }
    return 0;
}

- (double)sumSig:(int)type index:(int) index
{
    double up = 0;
    vDSP_mmulD((_theta + type * _dim), 1, _randomX[index], 1, &up, 1, 1, 1, _dim);
//    for (int i = 0; i < _dim; i++) {
//        up += _theta[type * _dim + i] * _randomX[index][i];
//    }
    up += _bias[type];
    
    double *down = malloc(sizeof(double) * _kType);
    double maxNum = 0;
    double sum = 0;
    vDSP_mmulD(_theta, 1, _randomX[index], 1, down, 1, _kType, 1, _dim);
    vDSP_vaddD(down, 1, _bias, 1, down, 1, _kType);
    vDSP_maxvD(down, 1, &maxNum, _kType);
    
    for (int i = 0; i < _kType; i++) {
        printf("%d:%f ",i,down[i]);
        down[i] -= maxNum;
        sum += expl(down[i]);
    }
    printf("\n");
 
    if (down != NULL) {
        free(down);
        down = NULL;
    }
    
    return expl(up - maxNum) / sum;
}

- (double *)fderivative:(int)type
{
    double *outP = malloc(sizeof(double) * _dim);
    double fillNum = 0.0f;
    vDSP_vfillD(&fillNum, outP, 1, _dim);
    
    double *inner = malloc(sizeof(double) * _dim);
    for (int i = 0; i < _randSize; i++) {
        long double sig = [self sumSig:type index:i];
        int ind = [self indicator:_randomY[i] var:type];
        double loss = -_descentRate * (ind - sig) / _randSize;
        _bias[type] += loss * _randSize;
        vDSP_vsmulD(_randomX[i], 1, &loss, inner, 1, _dim);
        vDSP_vaddD(outP, 1, inner, 1, outP, 1, _dim);
    }
    if (inner != NULL) {
        free(inner);
        inner = NULL;
    }
    
    // weight decay
//    double *decay = malloc(sizeof(double) * _dim);
//    double weight = 1e-4;
//    vDSP_vsmulD((_theta + type * _dim), 1, &weight, decay, 1, _dim);
//    vDSP_vaddD(outP, 1, decay, 1, outP, 1, _dim);
    
    return outP;
}

- (void)train
{
    _randomX = malloc(sizeof(double) * _randSize);
    _randomY = malloc(sizeof(int) * _randSize);
    for (int i = 0; i < _iterNum; i++) {
        [self randomPick:_trainNum];
        for (int j = 0; j < _kType; j++) {
            double *newTheta = [self fderivative:j];
            vDSP_vsmulD(newTheta, 1, &_descentRate, newTheta, 1, _dim);
            vDSP_vsubD(newTheta, 1, (_theta + j * _dim), 1, (_theta + j * _dim), 1, _dim);
//            for (int m = 0; m < _dim; m++) {
//                _theta[j * _dim + m] = _theta[j * _dim + m] - _descentRate * newTheta[m];
//            }
            if (newTheta != NULL) {
                free(newTheta);
                newTheta = NULL;
            }
        }
    }
}
*/
- (void)saveTrainDataToDisk
{
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *thetaPath = [[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) objectAtIndex:0] stringByAppendingString:@"/Theta.txt"];
//    NSLog(@"%@",thetaPath);
    NSData *data = [NSData dataWithBytes:_theta length:sizeof(double) *  _dim * _kType];
    [fileManager createFileAtPath:thetaPath contents:data attributes:nil];
    
    NSString *biasPath = [[NSSearchPathForDirectoriesInDomains(NSCachesDirectory, NSUserDomainMask, YES) objectAtIndex:0] stringByAppendingString:@"/bias.txt"];
    data = [NSData dataWithBytes:_bias length:sizeof(double) * _kType];
    [fileManager createFileAtPath:biasPath contents:data attributes:nil];
}

- (void)testModel
{
    [self randomPick:_trainNum];
    double correct = 0;
    for (int i = 0; i < _randSize; i++) {
        int pred = [self predict:_randomX[i]];
        if (pred == _randomY[i]) {
            correct++;
        }
    }
    printf("%f%%\n", correct/_randSize * 100.0);
}

- (int)predict:(double *)image
{
    double maxNum = 0;
    vDSP_Length label = 0;
    double *index = malloc(sizeof(double) * _kType);
//    vDSP_mmulD(_theta, 1, image, 1, index, 1, _kType, 1, _dim);
//    double *input = [_rnn forwardPropagation:image];
    double *input = [_lstm forwardPropagation:image];
    vDSP_mmulD(_theta, 1, input, 1, index, 1, _kType, 1, _dim);
    vDSP_vaddD(index, 1, _bias, 1, index, 1, _kType);
    vDSP_maxviD(index, 1, &maxNum, &label, _kType);
//    if (input != NULL) {
//        free(input);
//        input = NULL;
//    }
    return (int)label;
}

- (int)predict:(double *)image withOldTheta:(double *)theta andBias:(double *)bias
{
    double maxNum = 0;
    vDSP_Length label = 0;
    double *index = malloc(sizeof(double) * _kType);
    vDSP_mmulD(theta, 1, image, 1, index, 1, _kType, 1, _dim);
//    if (!_cnn) {
//        _cnn = [[MLCnn alloc] initWithFilters:@[@[@5,@5,@10],
//                                                @[@5,@5,@20]] fullConnectSize:_dim row:28 col:28];
//    }
//    double *input = [_cnn filterImage:image];
//    vDSP_mmulD(_theta, 1, input, 1, index, 1, _kType, 1, _dim);
    vDSP_vaddD(index, 1, bias, 1, index, 1, _kType);
    vDSP_maxviD(index, 1, &maxNum, &label, _kType);
//    if (input != NULL) {
//        free(input);
//        input = NULL;
//    }
    return (int)label;
}

- (void)updateModel:(double *)image label:(int)label
{
    double *index = malloc(sizeof(double) * _kType);
    // calculate wx+b
//    double *input = [_rnn forwardPropagation:image];
    double *input = [_lstm forwardPropagation:image];
    vDSP_mmulD(_theta, 1, input, 1, index, 1, _kType, 1, _dim);
    vDSP_vaddD(index, 1, _bias, 1, index, 1, _kType);
    // calulate exp(wx+b) / sum(exp(wx+b))
    index = [self MaxPro:index];
    for (int i = 0; i < _kType; i++) {
        double delta;
        if (i != label) {
            delta = 0.0 - index[i];
        }
        else
        {
            delta = 1.0 - index[i];
        }
        _bias[i] += _descentRate * delta;
        double loss = _descentRate * delta;
        double *decay = malloc(sizeof(double) * _dim);
        vDSP_vsmulD(image, 1, &loss, decay, 1, _dim);
        double *backLoss = malloc(sizeof(double) * _dim);
        vDSP_vsmulD((_theta + i * _dim), 1, &loss, backLoss, 1, _dim);
//        [_rnn backPropagation:backLoss];
        [_lstm backPropagation:backLoss];
        vDSP_vaddD((_theta + i * _dim), 1, decay, 1, (_theta + i * _dim), 1, _dim);
        if (decay != NULL) {
            free(decay);
            decay = NULL;
        }
    }
    if (index != NULL) {
        free(index);
        index = NULL;
    }
}

@end
