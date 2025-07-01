# *Creating and Testing Custom Layers in PyTorch.*

## *Goal*

The goal of this project is understanding the working of the following layers. 
* Conv2d
* Pooling
* Fully Connected
  
Then reimplementing them in pytorch from scratch to get a good grasp of the fundamentals of Pytorch and Numpy.

In order to test the models created using the above layers, I decided to use a sub portion of the **MNIST** dataset. 

To make it a little interesting, I decided to train 2 types of models,
1) Using only the convolution and pooling layers (although it does use a single fully connected layer in the end for the outputs)
2) Using only the fully connected layers ie a multi layer perceptron model.

I also structured these models to have _somewhat_ similar number of parameters.
The MLP has 12730 total learnable parameters.
The CNN 11146 total learnable parameters.

Note: ``` I did not implement the backpropogation algorithm from scratch, that remains for a future project :), I used the torch autograd for this one. ```

### Fully Connected Layer.
#### Implementation

![image](https://github.com/user-attachments/assets/f56dc188-e6a6-4c81-bb24-567af3e65d3b)

#### Explanation
* To create any custom model or layer in pytorch, it seems that I have to subclass the nn.Module.
* I created the weights and the bias, though I could have included the bias term as the 0th term in the weight matrix itself, I decided to go with AndrewNG way. When I create the weights using the nn.Parameter
function, it automatically includes it in the models parameters and its requires_grad is set to True by default. Creating with only the torh.tensor, the gradients wont be automatically tracked.
* For the activation function, I decided that I would let the user pick a torch defined actiation function but also a *custom one* too as the point here was to learn new stuff.
* In the forward method, I basically computed the matrix multiplication of the weight matrix with the input and added the bias term in the end.
``` WX + b ```
* As I have given 2 ways to use the activation function
    1) Function predefined by torch.
    2) Function defined by user.
I had to make sure what type the function was.
``` isinstance(self.activation, type) ``` This line checks whether the self.activation is a class. And if it is, I use ```issubclass(self.activation, torch.autograd.Function)``` to check if the class subclasses the torch.autograd.Function class. If this is the case, the activation is a user defined function else it is a torch defined function.

* After applying the activation, I return the result.

### Custom Activation Function

#### Implementation
![image](https://github.com/user-attachments/assets/eaf0233f-f311-4022-a3c8-95a5e75fa941)

#### Explanation
* To define a custom activation function, I have to subclass the torch.autograd.Function class.
* I define the forward method to compute the result from this function during the forward pass, the input is also saved for use during the back pass.
* The backward method is created to return the result of derivation of this activation function with respect to the output. The saved input is used for this computation.


### Custom Convolution Layer
#### Implementation
![image](https://github.com/user-attachments/assets/2b4e6f54-e2a8-4d29-b869-0842c38a8a08)
![image](https://github.com/user-attachments/assets/ce58231b-5321-4ca6-b7f3-88cd83393764)
#### Explanation
* As done before, I subclass the nn.Module class to create my own layer.
* I save the stride (how many rows and columns to skip while traversing), the padding (how many cells to add around the matrix), the kernal width and height (though I could have made it a square, I just went along with different), the number of out channels or filters, the activation function.
* I defined the weight matrix based on the input depth, filter size and the output depth(or number of filters) and the bias term (to be added once per filter).
* For the forward pass
*   1) I added the padding to the input matrix based on the ```padding``` from the user. For this, I used the inbuilt torch.nn.functional.pad function. Padding is passed 4 times to indicate that padding is to be added to heightm width, depth and front of the matrix.
    2) Torch has a really cool function to extract the patches or in other words it finds the regions where the kernal is to be multiplied and extracts these regions. This helps vectorize the code of traversing the kernal over the input matrix! ```patches = torch.nn.functional.unfold(x,kernel_size = (self.kH,self.kW),stride = self.stride)```
    3) In order to compute the output from traversing, I have to reshape the kernal, I can do this using the view function, which does not create a copy for reshaping which is efficient and works in this case.
    4) I use matmul to compute the result of the kernal traversing over the input, add the bias to each output dimension.
    5) To make sure the output dimensions are in the correct position, I use the permute function.
    6) I use the formula ``` Height new = (Height old - kernal height + 2*padding)//stride + 1, to find the height and width of the output to reshape it.
    7) I apply the activation function before returning the result.



### Custom Pooling Layer
#### Implementation
![image](https://github.com/user-attachments/assets/65407634-9833-439d-a67b-a2e0664cf437)
#### Explanation
* I subclass the nn.Module.
* I save the kernal size, stride, padding and mode. The mode here suggests max pooling or average pooling.
* In the forward method, I add the padding to the input matrix.
* Calculate the patches for the pooling operation. Except here, the pooling is applied layer by layer individually and not the whole 3d matrix.
* I reshaped the matrix in the required format, then used the max or the mean to compute the maximum or the average of the current patch.
* I used dim=2 because, the dimension corresponds to the elements in each patch layer.
* Finally I return a reshaped view of the result.

## Model 1 : ``` CNN ```
![image](https://github.com/user-attachments/assets/fd9e48c6-219f-4efe-9019-0fd7b2ea8015)

## Model 2 : ``` MLP ```
![image](https://github.com/user-attachments/assets/f13d6708-fb48-42c8-8082-ba8de62393fa)

# Results
### Dataset distribution:
![image](https://github.com/user-attachments/assets/0acf962a-2530-4713-abda-0f583fe83593)
### CNN Model Loss over 2000 epochs
![image](https://github.com/user-attachments/assets/0ef838f7-f5ec-4eda-ae45-e35097d12938)
### MLP Model Loss over 2000 epochs
![image](https://github.com/user-attachments/assets/c7b86908-8018-4030-9cf9-48edce4f754b)
### Comparison between the test loss of the 4 models
![image](https://github.com/user-attachments/assets/01a0fb32-7e92-45ef-8e10-0fe4f3d6c17c)

