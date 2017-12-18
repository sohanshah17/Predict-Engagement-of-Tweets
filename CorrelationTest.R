# load the library
library(mlbench)
library(caret)
# load the data
d=read.csv(file.choose(),stringsAsFactors=FALSE)
# calculate correlation matrix
correlationMatrix <- cor(d[,1:25])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (>0.5)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)