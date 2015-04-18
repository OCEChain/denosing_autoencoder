#ifndef READ_DATA
#define READ_DATA

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
using namespace std;
using namespace Eigen;

bool loadMnistData(MatrixXd &data,string szFileName)
{
    FILE *fp=fopen(szFileName.c_str(),"rb");
    if(!fp)
    {
        cout << "Could not Open " << szFileName << endl;
        return false;
    }
    unsigned int magic = 0;
    unsigned char temp;
    unsigned int numOfImages = 0;
    unsigned int rows;
    unsigned int cols;
    for(int i = 0;i < 4;i++)
    {
        if(feof(fp))
        {
            fclose(fp);
            return false;
        }
        fread(&temp,sizeof(char),1,fp);
        magic = magic << 8 | temp;
    }
    //printf("magic: %d\n",magic);
    for(int i = 0;i < 4;i++)
    {
        if(feof(fp))
        {
            fclose(fp);
            return false;
        }
        fread(&temp,sizeof(char),1,fp);
        numOfImages = numOfImages << 8 | temp;
    }
    //printf("numOfImages: %d\n",numOfImages);
    for(int i = 0;i < 4;i++)
    {
        if(feof(fp))
        {
            fclose(fp);
            return false;
        }
        fread(&temp,sizeof(char),1,fp);
        rows = rows << 8 | temp;
    }
    //printf("rows: %d\n",rows);
    for(int i = 0;i < 4;i++)
    {
        if(feof(fp))
        {
            fclose(fp);
            return false;
        }
        fread(&temp,sizeof(char),1,fp);
        cols = cols << 8 | temp;
    }
    //printf("cols: %d\n",cols);
    data.resize(rows*cols,numOfImages);
    for(int i = 0;i < (int)numOfImages;i++)
    {
        for(int j = 0; j < (int)(rows * cols); j++)
        {
            if(feof(fp))
            {
                cout << "Error reading file" << endl;
                fclose(fp);
                return false;
            }
            fread(&temp,sizeof(char),1,fp);
            data(j,i) = (temp / 255.0);
        }
    }
    fclose(fp);
    return true;
}


bool loadMnistLabels(MatrixXi &labels,string szFileName)
{
    FILE *fp=fopen(szFileName.c_str(),"rb");
    if(!fp)
    {
        cout << "Could not Open " << szFileName << endl;
        return false;
    }
    unsigned int magic = 0;
    unsigned char temp;
    unsigned int numOfLabels = 0;
    for(int i = 0;i < 4;i++)
    {
        if(feof(fp))
        {
            fclose(fp);
            return false;
        }
        fread(&temp,sizeof(char),1,fp);
        magic = magic << 8 | temp;
    }
    for(int i = 0;i < 4;i++)
    {
        if(feof(fp))
        {
            fclose(fp);
            return false;
        }
        fread(&temp,sizeof(char),1,fp);
        numOfLabels = numOfLabels << 8 | temp;
    }
    //printf("numOfLabels: %d\n",numOfLabels);
    labels.resize(numOfLabels,1);
    for(int i = 0;i < (int)numOfLabels;i++)
    {
        if(feof(fp))
        {
            fclose(fp);
            return false;
        }
        fread(&temp,sizeof(char),1,fp);
        labels(i,0) = temp;
    }
    fclose(fp);
    return true;
}





#endif // READ_DATA


