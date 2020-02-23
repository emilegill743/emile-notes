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

        # Setting Seaborn styling

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

    # Empirical cumulative distribution functions (ECDF)

        # Example
            import numpy as np

            x = np.sort(df_swing['dem_share'])

            y = np.arange(1, len(x)+1) / len(x)

            _ = plt.plot(x, y, marker='.', linestyle='none')
            _ = plt.xlabel('percent of vote for Obama')
            _ = plt.ylabel('ECDF')
            plt.margins(0.02)
            plt.show()

        # Function to calculate ECDF
            def ecdf(data):
                """Compute ECDF for a one-dimensional array of measurements."""
                # Number of data points: n
                n = len(data)

                # x-data for the ECDF: x
                x = np.sort(data)

                # y-data for the ECDF: y
                y = np.arange(1, n+1) / n

                return x, y

'''Quantitative Exploratory Data Analysis'''

    # Mean
        np.mean(dem_share_PA)

        # Heavily influenced by outliers

    # Median
        np.median(dem_share_PA)

        # Not influenced by outliers

    # Percentiles
        np.percentile([df_swing['dem_share'], [25, 50, 75])
        # Returns 25th, 50th, 75th percentiles

    # Box plot
        import matplotlib.pyplot as plt
        import seaborn as sns

        _ = sns.boxplot(x='east_west', y='dem_share',
                        data=df_all_states)
        _ = plt.xlabel('region')
        _ = plt.ylabel('percent of vote for Obama')
        plt.show()

    # Variance 
        # Mean squared distance of data from mean
        # Measure of spread

        np.var(dem_share_FL)

    # Standard Deviation
        # Square root of variance
        # Same units as quantity

        np.std(dem_share_FL)

    # Covariance and Pearson correlation coefficient

        # Scatter plot

            _ = plt.plot(total_votes/1000, dem_share,
                         marker='.', linestyle='none')
            _ = plt.xlabel('total votes (thousands)')
            _ = plt.ylabel('percent of vote for Obama')

        # Covariance

            # 1/n sum[ (x-x_mean)*(y-y_mean)]

            np.cov(x, y)
            # Returns covariance matrix
            # [0,0] --> variance in x
            # [1,1] --> variance in y
            # [0,1] and [1,0] --> covariance

        # Pearson correlation coefficient

            # p = covariance / (x_std * y_std)
            #   = variability due to codependence / independent variability
            # Dimensionless [-1, 1]

            np.corrcoef(x, y)
            # Returns 2x2 matrix
            # [0,0] and [1,1] neccesarily 1 (self-correlation)
            # [0,1] and [1,0] --> correlation coefficient
    
'''Thinking probabilistically - Discrete variables'''

    # Random number generators and hacker statistics

        # Hacker statistics - Uses simulated repeat measurements to compute probabilities
            
            # Determine how to simulate data
            # Simulate many times
            # Compute fraction of trials with outcome of interest


        # np.random module - suite of functions based on random number generation

        # Simulating a coin flip
        np.random.random() # draws random number [0,1], equal prob
        # Heads < 0.5
        # Tails >= 0.5
        
        # Bernouli trial - 
            # experiment with two options, success (True) and failure (False)

            def perform_bernoulli_trials(n, p):

                """Perform n Bernoulli trials with success probability p
                and return number of successes."""

                # Initialize number of successes: n_success
                n_success = 0

                # Perform trials
                for i in range(n):
                    # Choose random number between zero and one: random_number
                    random_number = np.random.random()

                    # If less than p, it's a success so add one to n_success
                    if random_number < p:
                        n_success += 1

                return n_success

        # Random number seed - integer fed into random number generating algorithm
        # Manually seed random number generator for reproducability
        np.random.seed()

        # Simulating 4 coin flips
            import numpy as np
            np.random.seed(42)
            random_numbers = np.random.random(size=4)

            heads = random_numbers < 0.5
            np.sum(heads)

        # Calculating the probability of 4 heads

            n_all_heads = 0 # Initialise no. trials with 4 heads

            # Simulate 10000 trials of 4 flips
            for _ in range(10000):
                heads = np.random.random(size=4) < 0.5
                n_heads = np.sum(heads)
                if n_heads == 4:
                    n_all_heads += 1

        # Predicting probability of defaults on a loan

            # 100 loans, p = 0.05 prob of default
            # Predicting probability of number of defaults from batch of 100 loans
            
            # Seed random number generator
            np.random.seed(42)

            # Initialize the number of defaults: n_defaults
            n_defaults = np.empty(1000)

            # Compute the number of defaults
            for i in range(1000):
                n_defaults[i] = perform_bernoulli_trials(100, 0.05)


            # Plot the histogram with default number of bins; label your axes
            _ = plt.hist(n_defaults, normed=True)
            _ = plt.xlabel('number of defaults out of 100 loans')
            _ = plt.ylabel('probability')

            # Show the plot
            plt.show()

        # Probability of bank losing money
            
            # Compute ECDF: x, y
            x, y = ecdf(n_defaults)

            # Plot the ECDF with labeled axes
            plt.plot(x,y, marker='.', linestyle='none')
            plt.xlabel('no. deafaults')
            plt.ylabel('ecdf')
            plt.show()

            # Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
            n_lose_money = np.sum(n_defaults >= 10)

            # Compute and print probability of losing money
            print('Probability of losing money =', n_lose_money / len(n_defaults))

    # Probability distributions

        # Probability mass function - set of probabilities of discrete outcomes
        # Probability distribution - mathematical description of outcomes

        # Discrete Uniform Distribution
            # The outcome of rolling a single fair die is discrete, uniformly distributed.

        # Binomial distribution
            # The number r of successes in n Bernoulli trials,
            # with probability p of success is Binomially distributed.

            samples = np.random.binomial(n, p, size=no_samples)

            # Binomial CDF
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                sns.set()

                x, y = ecdf(samples)
                _ = plt.plot(x, y, marker='.', linestyle='none')
                plt.margins(0.02)
                _ = plt.xlabel('number of successes')
                _ = plt.ylabel('CDF')
                plt.show()

        # Poisson distribution

            # Poisson process - The timing of the next event is completely independent
            #                   of when the previous event happened.

            # Examples of Poisson proccesses:
                # Natural births in a given hospital
                # Hits on a website in a given hour
                # Meteor strikes
                # Molecular collisions in a gas
                # Aviation accidents

            # The number r of arrivals of a Poisson process in a given time
            # with an average rate of l arrivals per interval, is Poisson distributed.

            # Poisson distribution is the limit of the Binomial distribution
            # for low prob of success p (rare events) and large number of trials n.
            
            samples = np.random.poisson(mean, size=no_samples)

            # Poisson CDF

                samples = np.random.poisson(6,size=10000)
                x, y = ecdf(samples)
                plt.margins(0.02)
                _ = plt.xlabel('number of successes')
                _ = plt.ylabel('CDF')
                plt.show()

    
'''Thinking probabilistically - Continuous variables'''
    
    # Probability density functions

        # Continuous analogue to the PMF.
        # Mathematical description of the relative likelihood of observing a value of a continous variable.
        # Consider areas under curve to represent probabilites, as it doesn't make sense to consider probability of a single value

    # Normal distribution

        # Describes a continous variable whose PDF has a single symmetric peak
        # Parameterised by mean, st. dev.

        samples = np.random.normal(mean, std, size=no_samples)

        # Checking normality of Michaelson data

            import numpy as np

            mean = np.mean(michaelson_speed_of_light)
            std = np.stsd(michaelson_speed_of_light)

            samples = np.random.normal(mean, std, size=10000)

            x, y = ecdf(michaelson_speed_of_light)
            x_theor, y_theor = ecdf(samples)

'''Introduction to Hypothesis Testing'''

    # Hypothesis Testing - Assessment of how reasonable the observed data are, assuming a hypothesis is true

        # Pipeline for hypothesis testing
            
            # CLearly state the null hypothesis
            # Define test statistic
            # Generate many sets of simulated data assuming the null hypothesis is true
            # Compute the test statistic for each simulated data set
            # p-value is fraction of simulated data sets for which test stat is at least as extreme as for the observed data

    # Permutation sampling
        # Comparing two probability distributions
        # Null hypothesis - two samples distributed exactly the same
        # Concatenate two samples and scramble data, as if they were exactly the same
        # Compare individual distributions to permutation sample distributions

        #Example

            def permutation_sample(data1, data2):
                """Generate a permutation sample from two data sets."""

                # Concatenate the data sets: data
                data = np.concatenate((data1, data2))

                # Permute the concatenated array: permuted_data
                permuted_data = np.random.permutation(data)

                # Split the permuted array into two: perm_sample_1, perm_sample_2
                perm_sample_1 = permuted_data[:len(data1)]
                perm_sample_2 = permuted_data[len(data1):]

                return perm_sample_1, perm_sample_2

            for i in range(0,50):
                # Generate permutation samples
                perm_sample_1, perm_sample_2 = permutation_sample(rain_june, rain_november)

                # Compute ECDFs
                x_1, y_1 = ecdf(perm_sample_1)
                x_2, y_2 = ecdf(perm_sample_2)

                # Plot ECDFs of permutation sample
                _ = plt.plot(x_1, y_1, marker='.', linestyle='none',
                            color='red', alpha=0.02)
                _ = plt.plot(x_2, y_2, marker='.', linestyle='none',
                            color='blue', alpha=0.02)

            # Create and plot ECDFs from original data
            x_1, y_1 = ecdf(rain_june)
            x_2, y_2 = ecdf(rain_november)
            _ = plt.plot(x_1, y_1, marker='.', linestyle='none', color='red')
            _ = plt.plot(x_2, y_2, marker='.', linestyle='none', color='blue')

            # Label axes, set margin, and show plot
            plt.margins(0.02)
            _ = plt.xlabel('monthly rainfall (mm)')
            _ = plt.ylabel('ECDF')
            plt.show()

            # Since ecdfs of original data do not overlap with the band of permutation sample ecdfs
            # we can conclude that the two datasets do not have identical distributions

    # Test statistics and p-values

        # Test statistic - A single number that can be computed form observed data and from data you simulate under the null hypothesis
        # Basis as comparison of two

        # Here, as we assume the two datasets are distributed equally, we would expect the means to be equal - i.e. their difference to be 0
        # Choose test statistic as difference between means

        # Permutation replicate - value of test statistic calculated from permutation sample
        np.mean(perm_sample_PA) - np.mean(perm_sample_OH)

        # We can simulate many permutation samples and calculate the permutation replicates
        # and then plot a histogram (PDF) of these values

        # p-value - The probability of obtaining a value of your test statistic at least as
        # extreme as what was observed, under assumption that null hypothesis is true
        # NOT probability of H0 being true

        # p-value can be calculated from area to right of observed value

        # Small p-value - statistically significantly different to H0

        # Example

            def draw_perm_reps(data_1, data_2, func, size=1):
                """Generate multiple permutation replicates."""

                # Initialize array of replicates: perm_replicates
                perm_replicates = np.empty(size)

                for i in range(size):
                    # Generate permutation sample
                    perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

                    # Compute the test statistic
                    perm_replicates[i] = func(perm_sample_1, perm_sample_2)

                return perm_replicates

            def diff_of_means(data_1, data_2):
                """Difference in means of two arrays."""

                # The difference of means of data_1, data_2: diff
                diff = np.mean(data_1) - np.mean(data_2)

                return diff

            # Compute difference of mean impact force from experiment: empirical_diff_means
            empirical_diff_means = diff_of_means(force_a, force_b)

            # Draw 10,000 permutation replicates: perm_replicates
            perm_replicates = draw_perm_reps(force_a, force_b,
                                            diff_of_means, size=10000)

            # Compute p-value: p
            p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

            # Print the result
            print('p-value =', p)

'''Hypothesis Test Examples'''

    # A/B testing

        # Used by organisations to see if a strategy change gives a better result
        # Generally H0 is that test statistic is impervious to the change
        # Low p-value indicates change in strategy lead to improvement in performance 

        # Null hypothesis - click through rate not affected by redesign

        import numpy as np

        # clickthrough_A, clickthrough_B: array of 1s, 0s

        # Function to calculate test statistic - clickthrough rate
        def diff_frac(data_A, data_B):
            frac_A = np.sum(data_A) / len(data_A)
            frac_B = np.sum(data_B) / len(data_B)
            return frac_B - frac_A

        # Calculate test statistic for observed datasets
        diff_frac_obs = diff_frac(clickthrough_A, clickthrough_B)

        # Permutation test of clickthrough
        perm_replicates = np.empty(10000)
        
        for i in range(10000):
            perm_replicates[i] = permutation_replicate(clickthrough_A,
                                                       clickthrough_B
                                                       diff_frac)
        
        p_value = np.sum(perm_replicates >= diff_frac_obs) / 10000


            





