################ LLMSR with API ################

# oscillation 1
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_gpt3.5
# python main.py --use_api True --api_model "gpt-4o" --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_gpt4o


# oscillation 2
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_numpy.txt --log_path ./logs/oscillator2_gpt3.5
# python main.py --use_api True --api_model "gpt-4o" --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_numpy.txt --log_path ./logs/oscillator2_gpt4o


# bacterial-growth
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name bactgrow --spec_path ./specs/specification_bactgrow_numpy.txt --log_path ./logs/bactgrow_gpt3.5
# python main.py --use_api True --api_model "gpt-4o" --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_numpy.txt --log_path ./logs/bactgrow_gpt4o


# stress-strain
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name stressstrain --spec_path ./specs/specification_stressstrain_numpy.txt --log_path ./logs/stressstrain_gpt3.5
# python main.py --use_api True --api_model "gpt-4o" --problem_name stressstrain --spec_path ./specs/specification_stressstrain_numpy.txt --log_path ./logs/stressstrain_gpt4o





################ LLMSR with LOCAL LLM ################

# oscillation 1
# python main.py --problem_name oscillator1 --spec_path ./specs/specification_oscillator1_numpy.txt --log_path ./logs/oscillator1_local


# oscillation 2
# python main.py --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_numpy.txt --log_path ./logs/oscillator2_local


# bacterial-growth
# python main.py --problem_name bactgrow --spec_path ./specs/specification_bactgrow_numpy.txt --log_path ./logs/bactgrow_local


# stress-strain
# python main.py --problem_name stressstrain --spec_path ./specs/specification_stressstrain_numpy.txt --log_path ./logs/stresstrain_local





################ EXAMPLE RUNS WITH TORCH OPTIMIZER ################


# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_torch.txt --log_path ./logs/oscillator2_gpt3.5_torch
# python main.py --problem_name oscillator2 --spec_path ./specs/specification_oscillator2_torch.txt --log_path ./logs/oscillator2_local_torch