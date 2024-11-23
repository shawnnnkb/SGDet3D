pip install -v -e .
cd packages/DFA3D
bash setup.sh 0
cd ../..
python packages/Voxelization/setup.py develop
python packages/Voxelization/setup_v2.py develop
# pip install spconv-cu113
# pip install cumm-cu113

# Question: A script repeatedly shows syntax errors during execution, but when copied and pasted into a newly created script, it runs without issues.
# Answer: The issue likely lies in the script's file format, which might differ between Windows and Unix systems. You can use vim to check the file format by entering :set ff in command mode.
# For Windows, the format should display as fileformat=dos.
# For Unix, it should display as fileformat=unix.
# To fix the issue, you can change the file format to Unix by entering :set ff=unix in vim. After saving the script, it should execute without errors.