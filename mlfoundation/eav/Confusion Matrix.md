## Confusion Matrix

- **True Positive** - If predicted positive and actual positive
- **True Negative** - If predicted negative and actual negative
- **False Positive** - If predicted positive but actual negative
- **False Negative** - If predicted negative but actual positive

![Confusion Matrix](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/5944c743_confusion/confusion.png)



- **False Positive(Type I error)** - Error of the first kind
- **False Negative(Type II error)** - Error of the second kind



Algorithms that minimizes Squared Error:

- Ordinary Least Squares(OLS) - Used n sklearn
- Gradient Descent

#### Using Squared over Absolute error

- Multiple line can fit in case of absolute error but only one line will fit in case of Squared Error
- Squared Error  makes implementation much easier: can be easily minimized by differentiating and equating to zero

#### R^2 answers the question

> How much of my change in  the output(y) is explained by change in my input(x)
>
> 0 < r^2 < 1