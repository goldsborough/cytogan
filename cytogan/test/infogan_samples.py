from cytogan.models import infogan

samples = infogan.sample_variables(20, 10, 3)
print(samples)
