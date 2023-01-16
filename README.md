# PixelwiseColorConstancy

### DNN code
The DNN codes are located in the code folder along with some sample images from our dataset.

Use file main.py in this folder to train the model.
Use file maintest.py in this folder to test the model.



### materials & image gernerator
The Mitsuba scene layout as well as the colors and illuminants spectra used along with some sample objects are located in the material folder
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
