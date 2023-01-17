# Object-based color constancy in a deep neural network

Color constancy refers to our capacity to see consistent colors under different illuminations. In computer vision and image processing, color constancy is often approached by explicit estimation of the scene's illumination, followed by an image correction. In contrast, color constancy in human vision is typically measured as the capacity to extract color information about objects and materials in a scene consistently throughout various illuminations, which goes beyond illumination estimation and might require some degree of scene and color understanding. Here we pursue an approach with Deep Neural Networks (DNNs) that tries to assign reflectances to individual objects in the scene. To circumvent the lack of massive ground truth datasets labeled with reflectances, we used computer graphics to render images. This study presents a model that recognizes colors in an image pixel by pixel under different illumination conditions.


Heidari-Gorji, H. & Gegenfurtner, K.R. (2023) Object-based color constancy in a deep neural network. Journal of the Optical Society of America A, in press. https://doi.org/10.1364/JOSAA.479451

## DNN code
The DNN codes are located in the code folder along with some sample images from our dataset.

Use file main.py in this folder to train the model.
Use file maintest.py in this folder to test the model.



## Materials & image gernerator

### Colors

For color we use  1600 Munsell colors glossy from: https://sites.uef.fi/spectral/munsell-colors-glossy-all-spectrofotometer-measured/

### illuminations
We used 17 different illumination spectra. You can find them in the material/illumination folder

### Spectral renderer engine
For generating datset we used Mitsuba spectral renderer engine
The Mitsuba scene layout is located in the material folder
To generate new images, Mitsuba must be installed from https://github.com/mitsuba-renderer/mitsuba

After that you should run this command in the command prompt:
```
mitsuba 
-DfileColor1=color/[colorname].csv                           \\ Color of first object in this scene layout
-DfileColor2=color/[colorname].csv                           \\ Color of first object in this scene layout
-DfileColor3=color/[colorname].csv                           \\ Color of first object in this scene layout
-DfileColor4=color/[colorname].csv                           \\ Color of statue in this scene layout
-DfileColor5=color/[colorname].csv                           \\ Color of table in this scene layout
-Dobject1=object/[objectname].obj                            \\ First object in this scene layout
-Dobject2=object/[objectname].obj                            \\ Second object in this scene layout
-Dobject3=object/[objectname].obj                            \\ Third object in this scene layout
-Dmaterial1=[diffuse/plastic/metal]                          \\ Material of first object
-Dmaterial2=[diffuse/plastic/metal]                          \\ Material of second object
-Dmaterial3=[diffuse/plastic/metal]                          \\ Material of third object
-Dmaterial4=[diffuse/plastic/metal]                          \\ Material of statue 
-Dmaterial5=[diffuse/plastic/metal]                          \\ Material of table 
-DglassMat=[diffuse/plastic/metal/glass]                     \\ Material of cups 
-Dfileillu=illumination/[illuminationfilename].csv           \\ illumination file name
room.xml
```

![alt text](https://github.com/haamedh/PixelwiseColorConstancy/blob/main/materials/room.png?raw=true)
