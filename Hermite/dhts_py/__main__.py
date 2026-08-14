from .tests.test_dht2_reconstruction import run_dht2_test

result = run_dht2_test()
print("DHTS Python self-check")
print(f"max reconstruction error: {result['max_abs_error']:.3e}")
print(f"mean squared error:       {result['mse']:.3e}")
