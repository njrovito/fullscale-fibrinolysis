import flatiron_tk as fe
import sys, os

def main():
    ans = input('Output directory is \'output.\' Ok? (Y/N): ')
    if ans == 'y' or ans == 'Y':
        output_dir = 'output/'
    else:
        output_dir = input('Input new output directory (include \'/\'): ')

    for file in os.listdir(output_dir):
        if file.endswith('.h5'):
            h5_group = file[:-3]
            file = output_dir + file
            pvd_file = file[:-3] + '.pvd'
            if file.__contains__('u.h5'):
                function_type = 'vector'
            else:
                function_type = 'scalar'
            fe.h5_to_pvd(file, h5_group, pvd_file, function_type, 'CG', 1, 'all')
    print('Done.')
    return

if __name__ == '__main__':
    main()