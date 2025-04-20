from MagneticMoment import calculate_moment
from InputParameters import load_params
from ActuationMatrix import construct_matrix
from OptimalPlacement import find_opt_positions
from OutputResults import output_results
from GeneratePoses import generate_poses


def main():
    try:
        params = load_params()
        print("Successfully loaded params from config file.")
    except Exception as e:
        print(f"Error: {e}")

    moment = calculate_moment(params.N, params.I, params.A)

    poses = generate_poses(params.M, params.l)

    U_matrix = construct_matrix(poses, moment, params.m_t, params.t)
    soln_x = find_opt_positions(params.M, params.K, U_matrix, poses, params.l)

    output_results(soln_x, params.K, poses)


if __name__ == "__main__":
    main()