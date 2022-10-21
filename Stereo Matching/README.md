# winner-take all stereo algorithm using the rank-transform 
- compute rank transform in 5x5 windows
- compute disparity maps on the rank-transformed images, aggregating the absolute differences of rank in 3x3 and 15x15 windows.
- compute error rates by counting the fraction of pixels that differs bymore than one disparity level from the ground truth
- For the 3x3 aggregation window above, compute matching confidence using the PKRN measure.
- Using the PRKN values, generate a disparity map containing the top 50% most confident pixels.
- Report the error rate of the sparse disparity map and the number of pixels that have been kept<br/>

![image](https://user-images.githubusercontent.com/35480902/197114283-a1e89acc-8a46-4e69-883f-b4e2bf28d7fa.png)
![image](https://user-images.githubusercontent.com/35480902/197114183-4d759887-ade4-45e3-9320-b1d11381fad9.png)<br/>
![image](https://user-images.githubusercontent.com/35480902/197113796-9db1c992-9f7b-498b-8377-9cc8b2fe560a.png)
![image](https://user-images.githubusercontent.com/35480902/197114025-3c51454e-4a4e-4522-9028-e4a2083b4c48.png)<br/>

Right Image:<br/>
![image](https://user-images.githubusercontent.com/35480902/197114590-dd039274-253a-403e-b25b-75844d3ee627.png)<br/>
Left Image:<br/>
![image](https://user-images.githubusercontent.com/35480902/197114654-6f475a1b-246a-4ba1-a4e0-359cdf66e2b1.png)<br/>

Ground Truth:<br/>
![image](https://user-images.githubusercontent.com/35480902/197114721-86112f93-aeea-498c-b6bc-02ffb9881cff.png)<br/>


