import os
from random import randint
from time import sleep


def main():
	print('\nStart saver-program\n')
	file_path = os.path.join(os.path.dirname(__file__).replace('programs', 'plots'), 'saver_result.txt')
	
	with open(file_path, 'w') as f:
		f.write('')
	for i in range(0, 10,):
		sleep(randint(5, 16))
		with open(file_path, 'a') as f:
			f.write('%d\n' % (10-randint(0, 10)))  # python will convert \n to os.linesep
		print('\nWritten %d\n' % i)
		
	print('\n\nProgram is done.\n\n')
	
main()
