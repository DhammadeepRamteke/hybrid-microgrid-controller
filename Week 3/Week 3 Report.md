# **Week 3 Report : Smart Controller Simulation & Analysis**

---

This document details the complete analysis performed during Week 3 of the project, corresponding to the "Smart Controller Logic & Simulation" milestone.

All code and analysis described here can be found in the Smart\_Controller.ipynb Jupyter Notebook. 

## **1\. Objective** 

* The goal of Week 2 was to train and save predictive models. The objective of Week 3 is to use those models to power a Smart Microgrid Controller. 

### **This involves :** 

* Loading the trained RandomForestRegressor models (.joblib files) and the scaler.joblib.   
* Using these models to generate a 24-hour-ahead forecast for load\_demand, solar\_output, and wind\_output.   
* Simulating an intelligent battery controller that uses these forecasts to make optimal decisions about charging, discharging, or interacting with the main grid.   
* Evaluating the controller's performance by analyzing battery usage, grid dependency, and overall cost-effectiveness. 

### **2\. Methodology & Code Analysis** 

#### **Step 2.1 : Loading Models and Data** 

* **Load Artifacts :** The notebook begins by loading the three .joblib models (for load, solar, and wind) and the scaler.joblib file created in Week 2\.   
* **Load Simulation Data :** The processed\_dataset.csv (the output from Week 1\) is loaded. This dataset contains the "ground truth" (actual) values that our simulation will be compared against. 

#### **Step 2.2 : Generating Forecasts** 

* Before the simulation can run, the controller needs a forecast.   
* The models are used to predict predicted\_load\_demand, predicted\_solar\_output, and predicted\_wind\_output for the entire dataset.   
* **From these, a new key feature is engineered :** predicted\_net\_energy (Predicted Solar \+ Predicted Wind \- Predicted Load). This single value tells the controller whether it should expect a surplus (positive) or deficit (negative) of energy in the next hour. 

#### **Step 2.3 : Smart Controller Simulation Logic** 

* This is the core of the notebook. A simulation is run hour-by-hour over the entire dataset to determine how the controller would behave. 

#### **Constants Defined :** 

* **BATTERY\_CAPACITY\_MWH :** 100.0 MWh   
* **MAX\_CHARGE\_RATE\_MW :** 50.0 MW 

#### **Simulation Loop :** 

* The controller's logic iterates through each hour with the following rules : 

#### **A. If predicted\_net\_energy \> 0 (Predicted Surplus) :** 

* The controller first attempts to charge the battery with the surplus power.   
* It calculates the charge\_amount, respecting the battery's remaining capacity and the 50 MW max charge rate.   
* The battery\_soc (State of Charge) is increased.   
* Any leftover surplus power after the battery is full is sold to the grid (a positive grid\_exchange). 

#### **B. If predicted\_net\_energy \< 0 (Predicted Deficit) :** 

* The controller first attempts to cover the deficit by discharging the battery.   
* It calculates the discharge\_amount, respecting the battery's available charge and the 50 MW max discharge rate.   
* The battery\_soc is decreased.   
* Any remaining deficit after the battery is empty is bought from the grid (a negative grid\_exchange). 

#### **C. Simulation Results & Visualization** 

* The notebook visualizes the simulation results to analyze the controller's performance. 

#### **1\) Battery State of Charge (SoC)** 

* The battery\_soc column from the simulation DataFrame (df\_sim) is plotted over time.   
  * **Analysis :** The plot above shows the battery's state of charge over the course of the simulation.   
  * **Insight :** The battery exhibits a clear and logical cyclical pattern: it charges during the day when the predicted\_net\_energy (driven by solar) is high and discharges at night to meet the load. This confirms the controller is successfully storing and deploying energy to manage supply and demand. 

![](https://github.com/DhammadeepRamteke/hybrid-microgrid-controller/blob/3837aaa6d6f2dc13798c40194677ff52d9f3926d/Week%203/battery_soc.png)
Battery State of Charge (SOC) During Simulation

#### **2\) Grid Exchange (Time-Series)** 

* The grid\_exchange column, representing power bought (negative) or sold (positive), is plotted over time.   
  * **Analysis :** This plot shows the microgrid's interaction with the main power grid.   
  * **Insight :** For long stretches, the grid\_exchange is zero, indicating the microgrid is self-sufficient. The battery is successfully absorbing all surplus and covering all deficits. Only during extreme events does the controller need to buy or sell power. 

![](https://github.com/DhammadeepRamteke/hybrid-microgrid-controller/blob/3837aaa6d6f2dc13798c40194677ff52d9f3926d/Week%203/grid_exchange.png)
Grid Exchange Over Time

#### **3\) Grid Exchange (Distribution)** 

* A histogram of the grid\_exchange column shows the frequency of different interaction levels.   
  * **Analysis :** This histogram is the most important result. It shows the frequency of all grid\_exchange values from the simulation.   
  * **Key Insight :** There is a massive spike at 0 MWh. This visually confirms that for the vast majority of the time, the smart controller was able to operate the microgrid in "island mode"—perfectly balancing the internal load, generation, and battery storage without any help from the main grid. 

![](https://github.com/DhammadeepRamteke/hybrid-microgrid-controller/blob/3837aaa6d6f2dc13798c40194677ff52d9f3926d/Week%203/grid_histogram.png)
Distribution of Grid Exchange

#### **D. Final Performance Metrics** 

* To quantify the controller's success, the simulation calculates the total energy bought/sold and compares it to a baseline "dumb" system (one with no battery).   
  * **Total Energy Bought (Smart Controller) :** 42,042 MWh   
  * **Total Energy Sold (Smart Controller) :** 52,656 MWh   
  * **Total Battery Cycles (Wear) :** 113.88   
  * **Total Energy that WOULD be Bought (No Battery) :** 117,143 MWh   
    (Calculated by summing the actual\_net\_energy during all deficit hours) 
