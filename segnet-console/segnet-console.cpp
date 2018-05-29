/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "segNet.h"
#include <QImage>
#include "loadImage.h"
#include "commandLine.h"
#include "cudaMappedMemory.h"

#include <sys/time.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <unordered_set>
#include <unordered_map>
uint64_t current_timestamp() {
  struct timeval te; 
  gettimeofday(&te, NULL); // get current time
  return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

// this map is obtained from
// https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py#L72
// and cityscapes-labels.txt
std::unordered_map<uint8_t, uint8_t> gt2pred;
void initGt2PredMap() {
  gt2pred = {
    {0, 0},
    {1, 0},
    {2, 0},
    {3, 0},
    {4, 0},
    {5, 1},
    {6, 2},
    {7, 3},
    {8, 4},
    {9, 5},
    {10, 0},
    {11, 6},
    {12, 7},
    {13, 8},
    {14, 9},
    {15, 10},
    {16, 10},
    {17, 11},
    {18, 0},
    {19, 12},
    {20, 13},
    {21, 14},
    {22, 15},
    {23, 16},
    {24, 17},
    {25, 0},
    {26, 18},
    {27, 19},
    {28, 19},
    {29, 0},
    {30, 0},
    {31, 0},
    {32, 21},
    {33, 21}
  };

}

void saveGrayImg(char * filename, char * preStr,  uint8_t * data, int width, int height) {
  char outFilename [200];
  memset(outFilename, 0, sizeof(outFilename));
  memcpy(outFilename, preStr, sizeof(preStr));
  strcat(outFilename, filename);

  int numPixels = width * height;
  QImage img (width, height, QImage::Format_Indexed8);
  for (int y = 0; y < height; y++) {
    memcpy(img.scanLine(y), data + y * width, width);
  }
  QVector<QRgb> colorTable;
  for (unsigned int i =0; i < 32; i++) {
    colorTable.push_back(qRgb(i*8, i*8, i*8));
  }
  for (unsigned int i = 32; i < 256; i++) {
    colorTable.push_back(qRgb(i*8, i*8, i*8));
  }
  img.setColorTable(colorTable);
  img.save(outFilename );
}


void  updateIoU(uint64_t ** iouCounter,  QImage * qImg, uint8_t* prediction, int numClasses, char*filename) {
  uchar * byteP = qImg->bits();
  int width = qImg->size().width() , height = qImg->size().height();
  int numBytes = width * height;

  uint8_t * classIDConverted = (uint8_t *) malloc(numBytes);
  for (int i = 0; i< numBytes; i++) {
    *(classIDConverted+i) = gt2pred[*(uint8_t*)(byteP+i)];   
  }
  //saveGrayImg(filename, "gt_", classIDConverted, width, height );
  //saveGrayImg(filename, "pred_", prediction, width, height);
  
  std::unordered_set<uint8_t> classIds;
  std::unordered_set<uint8_t> predIds;
  for (int i = 0; i < numBytes; i++) {
    uint8_t gtPixel = *(classIDConverted+i);
    uint8_t predPixel = *(prediction+i);
    classIds.insert(gtPixel);
    predIds.insert(predPixel);
    if (gtPixel > numClasses-1 ) {

      continue;
    }
    if (gtPixel == predPixel )
      iouCounter[gtPixel][gtPixel] ++;
    else 
      iouCounter[gtPixel][predPixel] ++;
  }
  printf("class id include : ");
  for (auto item : classIds) {
    printf("%hhu, ", item);
  }
  printf("\n");
  printf("pred id include : ");
  for (auto item : predIds) {
    printf("%hhu, ", item);
  }
  printf("\n");

}
float mIoU(uint64_t ** iouCounter, int numClasses) {
  float miou = 0;
  int miouDenominator = numClasses-1;
  for (int c = 1; c < numClasses; c++) {
    float numerator = (float)iouCounter[c][c];
    float denominator = 0;
    for (int j = 1; j < numClasses; j++) {
      denominator = denominator + iouCounter[c][j] + iouCounter[j][c]; 
    }
    denominator -= numerator;
    if (denominator == 0) {
      miouDenominator--;
      continue;
    }
    miou = miou + numerator/denominator;
					  
  }
  miou /= miouDenominator;
  return miou;
}

// filename: local file name, not path
void drawAndStore(uint8_t * classMap, float * outcpu, segNet * net, int height, int width, char * filename) {
  if (net == NULL) {
    printf("Error: net is null");
    return;
  }
  net->DrawInColor(classMap, outcpu, height, width);
  char outFilename [200];
  memset(outFilename, 0, sizeof(outFilename));
  memcpy(outFilename, "networkOut_", sizeof("networkOut_"));
  strcat(outFilename, filename);
  if( !saveImageRGBA(outFilename, (float4*)outcpu, width, height) )
    printf("segnet-console:  failed to save output image to '%s'\n", outFilename);
  else
    printf("segnet-console:  completed saving '%s'\n", outFilename);

}

// main entry point
int main( int argc, char** argv )
{
  printf("segnet-console\n  args (%i):  ", argc);
	
  for( int i=0; i < argc; i++ )
    printf("%i [%s]  ", i, argv[i]);
		
  printf("\n\n");

	
  // retrieve filename arguments
  if( argc < 2 ) {
    printf("segnet-console:   input image folder name required\n");
    return 0;
  }

  if( argc < 3 )  {
    printf("segnet-console:   ground truth filename required\n");
    return 0;
  }

  initGt2PredMap();
  // create the segNet from pretrained or custom model by parsing the command line
  segNet* net = segNet::Create(argc, argv);

  if( !net ) {
    printf("segnet-console:   failed to initialize segnet\n");
    return 0;
  }
	
  // enable layer timings for the console application
  net->EnableProfiler();

  char imgFilename[200];
  char gtFilename[200];

  DIR * imgDirP = opendir(argv[1]);
  dirent * imgDirent = NULL;


  // iou init
  int numClasses = net->GetNumClasses();
  uint64_t ** iouCounter = (uint64_t **)malloc(numClasses * sizeof(uint64_t *));
  for (int i = 0; i!= numClasses; i++) {
    iouCounter[i] = (uint64_t *) malloc(numClasses * sizeof(uint64_t));
    memset(iouCounter[i], 0, numClasses * sizeof(uint64_t));
  }
  
  int numImages = 0;
  while (imgDirP) {
    errno = 0;
    if ((imgDirent = readdir(imgDirP)) == NULL) {
      closedir(imgDirP);
      if (errno == 0) {
	printf("img folder not found\n");
      } else {
	printf("img folder read error\n");
      }
      break;
    }
    char * imgName = imgDirent->d_name;
    if (strlen(imgName)  < 4) continue;

    size_t preStrLen = strlen(imgName) - 15; // rm leftImg8bit.png
    char * preStr = (char *) malloc(preStrLen+1);
    memcpy(preStr, imgName, preStrLen);
    preStr[preStrLen] = 0;
    printf("\npreStr is %s\n", preStr);
    memset(imgFilename, 0, sizeof(imgFilename));
    memset(gtFilename, 0, sizeof(gtFilename));
  
    strcat(imgFilename, argv[1]);
    strcat(imgFilename, "/");
    strcat(imgFilename,  imgName);
    strcat(gtFilename, argv[2]);
    strcat(gtFilename, "/");
    strcat(gtFilename, preStr);
    strcat(gtFilename, "gtFine_labelIds.png");

    printf("img #%d, source file name is %s, ground truth file name is %s\n", numImages, imgFilename, gtFilename );


    // load image from file on disk
    float* imgCPU    = NULL;
    float* imgCUDA   = NULL;
    int    imgWidth  = 0;
    int    imgHeight = 0;
    uint8_t* imgGT     = NULL;
    
    if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) ) {
      printf("failed to load image '%s'\n", imgFilename);
      return 0;
    }

    QImage qImg;
    if(!qImg.load(gtFilename)) {
      printf("failed to load ground truth image\n");
      return 0;
    }
    printf("Bits per pixel in gt is %d, isGrayScale is %d\n", qImg.depth(), qImg.isGrayscale());
    
    // allocate output image
    float* outCPU  = NULL;
    uint8_t* outCUDA = NULL;

    if( !cudaAllocMapped((void**)&outCPU, (void**)&outCUDA, imgWidth * imgHeight * sizeof(float) * 4) )      {
	printf("segnet-console:  failed to allocate CUDA memory for output image (%ix%i)\n", imgWidth, imgHeight);
	return 0;
      }

    printf("segnet-console:  beginning processing forward (%zu)\n", current_timestamp());
    
    // set alpha blending value for classes that don't explicitly already have an alpha	
    net->SetGlobalAlpha(120);
   
    // process image overlay
    if( !net->ForwardResult(imgCUDA, &outCUDA, imgWidth, imgHeight, "void") ) {
      printf("segnet-console:  failed to process segmentation forward.\n");
      return 0;
    }
    printf("network out address %p\n", outCUDA);
    printf("segnet-console:  finished forward propagation  (%zu)\n", current_timestamp());
    //drawAndStore(outCUDA, outCPU, net, imgHeight, imgWidth, imgName);
      
    // compute mIoU
    updateIoU( iouCounter,  & qImg, outCUDA, numClasses, imgName);
    printf("Finish updating IOU table");
    numImages++;
    if (numImages % 1 == 0)
      printf("\nSummary: mIoU is %.4f over %d images\n", mIoU(iouCounter, numClasses), numImages );
    /*
    // save output image
    if( !saveImageRGBA(outFilename, (float4*)outCPU, imgWidth, imgHeight) )
      printf("segnet-console:  failed to save output image to '%s'\n", outFilename);
    else
      printf("segnet-console:  completed saving '%s'\n", outFilename);
    */
   
    CUDA(cudaFreeHost(imgCPU));
    CUDA(cudaFreeHost(outCPU));

  }
  printf("\nSummary: mIoU is %.4f over %d images\n", mIoU(iouCounter, numClasses), numImages );
  delete net;
  return 0;
}
