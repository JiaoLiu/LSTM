//
//  MLLoadMNIST.h
//  MNIST
//
//  Created by Jiao Liu on 9/23/16.
//  Copyright © 2016 ChangHong. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface MLLoadMNIST : NSObject

- (double **)readImageData:(const char *)filePath;
- (int *)readLabelData:(const char *)filePath;

@end
