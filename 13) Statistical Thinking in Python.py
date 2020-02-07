#####STATISTICAL THINKING IN PYTHON#####

'''Graphical Exploratory Data Analysis'''

    # Plotting a histogram
        import matplotlib.pyplot as plt
        _ = plt.hist(df_swing['dem_share'])
        _ = plt.xlabel('percent of vote for Obama')
        _ = plt.ylabel('number of counties')
        plt.show()

        # plt.hist() returns 3 arrays we are not interested in
        # using dummy variables _ lets us just focus on the plot

        # Setting bins

            # Specifying bin edges
            bin_edges = [0, 10, 20, 30, 40, 50,
                        60, 70, 80, 90, 100]
            _ = plt.hist(df_swing['dem_share'], bins=bin_edges)
            plt.show()

            # Setting number of bins
            _ = plt.hist(df_swing['dem_share'], bins=20)
            plt.show()

            # Binning bias - the same data may be interpreted differently depending on choice of bins

        #Setting Seaborn styling
        import seaborn as sns
        sns.set()
        _ = plt.hist(df_swing['dem_share'])
        _ = plt.xlabel('percentage of vote for Obama')
        _ = plt.ylabel('number of counties')
        plt.show()

    # Bee swarm plots

        _ = sns.swarmplot(x='state', y='dem_share', data=df_swing)
        _ = plt.xlabel('state')
        _ = plt.ylabel('percent of vote for Obama')
        plt.show()
