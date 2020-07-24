## Gas object

This object is a C struct that has the input and output parameters of a gas functions. A C struct was used for this object to save processing time. Some output parameters include the Cross sections, and the energy levels. Some of the input parameters include the final electron energy and the energy step. 

## Gasmix object 

This object is used to call the correct gas functions. This object consists mainly of an array of Gas objects and a dictionary of extra parameters. The extra parameters are used for special gas functions that require extra input. For example, the xenon MERT function. 
