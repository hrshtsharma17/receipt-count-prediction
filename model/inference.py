import os
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Inference:
    def __init__(self, path, data_path) -> None:
        """
        self.inf_stack caches results from older queries
        and intermediate results
        """
        self.model = load_model(path)
        self.inf_stack = pd.read_csv(data_path)
        self.inf_stack['Date'] = pd.to_datetime(self.inf_stack['# Date'])
        self.inf_stack.drop(columns=["# Date"], inplace=True)

        self.n_steps = 120
        self.n_features = 1
        self.start_date = self.inf_stack.iloc[0]["Date"]
        self.end_date = self.inf_stack.iloc[-1]["Date"]
    
    def _predict(self, input):
        input = input.reshape((1, self.n_steps, self.n_features))
        return self.model.predict(input, verbose=0)
        
    def predict_n_steps(self, xinput, steps):
        """
        input:
            xinput: (n_steps,)
            steps: number of next predictions required

            We can use self.inf data directly too but this way we can pass explicit data

        output: 
            numpy array, list containing the prediction for next nsteps
        """

        out = [0]*steps
        for i in range(steps):
            min_i, max_i = min(xinput), max(xinput) # as max and min is shifting we need to do this inside loop
            x_input = (xinput - min_i)/(max_i-min_i)
            next_y = self._predict(x_input)
            ypred = next_y[0][0]*(max_i-min_i) + min_i

            out[i] = ypred
            xinput = np.append(xinput[1:], [ypred])
        
        return np.array(out)
    
    def predict_by_date_range(self, start_date, end_date):
        if start_date < self.start_date or end_date < start_date:
            return "Sorry! please provide a month from 2021 onwards"
        else:
            start_index = (start_date-self.start_date).days
            end_index = (end_date-self.start_date).days
            current_timestamps = len(self.inf_stack)
            if end_index < current_timestamps:
                return self.inf_stack.iloc[start_index:end_index+1]["Receipt_Count"]
            else:
                steps = end_index - current_timestamps + 1
                xinput = np.array([self.inf_stack[-self.n_steps:]["Receipt_Count"]])
                xinput = xinput.reshape((self.n_steps,))

                out = self.predict_n_steps(xinput, steps)
                next_date, i = self.end_date, 0
                while steps:
                    next_date = next_date + timedelta(days=1)
                    next_row = pd.DataFrame({"Date": next_date,"Receipt_Count":out[i]}, index=[0])
                    self.inf_stack = pd.concat([self.inf_stack, next_row], ignore_index=True)
                    i+=1
                    steps-=1
                
                self.end_date = self.inf_stack.iloc[-1]["Date"]
                return self.inf_stack.iloc[start_index:end_index+1]["Receipt_Count"]
        
    def predict_by_month(self, month_string):
        """
        Input:
            month_string: string, format: month-year ex.12-2022
        Output:
            Value of prediction for the month, float
        """
        month, year = map(int, month_string.split("-"))
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year, month, 31)
        else:
            next_month = start_date + timedelta(days=31)
            end_date = next_month - timedelta(days=next_month.day)
        
        result = self.predict_by_date_range(start_date, end_date)
        return sum(result) if type(result)!=str else result


def generate_monthly_predict_plot():
    past_st_date, past_end_date = "2021-01-1", "2021-12-31"
    start_date, end_date = "2022-01-1", "2022-12-31"

    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    past_st_date = datetime.strptime(past_st_date, '%Y-%m-%d')
    past_end_date = datetime.strptime(past_end_date, '%Y-%m-%d')

    output_folder = "model/artifacts_2023-10-08_20-11-39"
    inf = Inference(f"{output_folder}/best_model.h5", 'data/data_daily.csv')
    xinput = inf.predict_by_date_range(past_st_date, past_end_date)
    result = inf.predict_by_date_range(start_date, end_date)

    # Create an array of month values (1-12) corresponding to each day in the data
    monthwise_day_dist = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for i in range(1, 13):
        monthwise_day_dist[i] = monthwise_day_dist[i]+monthwise_day_dist[i-1]

    monthwise_past, monthwise_future = [0]*12, [0]*12
    for i in range(1, 13):
        st, end = monthwise_day_dist[i-1], monthwise_day_dist[i]
        monthwise_past[i-1] = sum(xinput[st:end])
        monthwise_future[i-1] = sum(result[st:end])

    fig, ax = plt.subplots()

    xstart_date, ystart_date = datetime(2021, 1, 1), datetime(2022, 1, 1)
    xdate_range = [xstart_date + timedelta(days=i*30) for i in range(12)]
    ydate_range = [ystart_date + timedelta(days=i*30) for i in range(12)]

    ax.plot(xdate_range, monthwise_past, label='Past Data', c='orange')
    ax.plot(ydate_range, monthwise_future, label='forecast values', c='blue')
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))  # Display every 2 months
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Month and Year format

    ax.set_xlabel('Months')
    ax.set_ylabel('Approximate Receipt Count')
    ax.set_title(f'Forecast Graph - Next {str(len(monthwise_future))} months')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'forecast_plot.png'))
    plt.close()

if __name__=="__main__":
    generate_monthly_predict_plot()