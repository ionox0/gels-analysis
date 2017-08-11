# Automated Gel Electrophoresis Classification
### CU Medical Center, Worgall lab

In this project I attempt to perform a classification task on segmented gel electrophoresis lanes using various image processing and classification techniques. 

The images are samples from individuals suspected of having Multiple Myeloma, which is indicated by the presence of additional banding patterns one of several regions along the lane. 

Initial work centered around a manual classification process, using a web application to hand label individual lanes:

[Webapp](gels-analysis.herokuapp.com)

See [training results](https://github.com/ionox0/gels-analysis/blob/master/misc/Training%20Attempt%201.pdf) for initial accuracy measurements.

Current work is focused on a fully-automated process by which whole gel images are uploaded, and individual lanes are detected and matched with labels from an excel file.

