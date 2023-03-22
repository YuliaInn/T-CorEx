## Finding optimal number of hidden layers

input file is in https://drive.google.com/file/d/1Jal4dVZOX-wCGWBUnbBwK6fStYqMcG7T/view?usp=share_link

here is the command to find optimal number* of HLs using cpu

```{bash}
python test/testNLL.py path_to_inputfile
```

Here is the command to find a number* of HLs using GPU

```{bash}
python NLL_pick/NLL_pick.py path_to_inputfile
```

* I hard coded the range on HLs but we can change it later