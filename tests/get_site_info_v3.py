import sys
import time

from scipy.spatial import ConvexHull
from pymatgen.io.vasp.outputs import *
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import *
from mapping_sublattice.local_bcc_sublattice.find_all_bcc_tet_oct.fcc_2_bcc.process_fcc2bcc_data import get_tet_oct_detail_info
from pymatgen.analysis import bond_valence

class HiddenPrints:
    '''
    class to reduce the output lines
    '''

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def non_elements(struct, sp_anion):
    '''
    struct : structure object from Pymatgen
    sp : the mobile specie
    returns the structure with all of the mobile specie (Li) removed (错误的)
    返回的结构只包含阴离子框架 (Cl)
    '''
    # num_li = struct.species.count(Element(sp))
    species = list(set(struct.species))
    species.remove(Element(sp_anion))
    stripped = struct.copy()
    stripped.remove_species(species)
    stripped = stripped.get_sorted_structure(reverse=True)
    # note reverse=True好像是反向排列
    return stripped


def site_env(coord, struct, sp="Li", sp_anion="Cl", envtype='both'):
    '''
    coord : Fractional coordinate of the target atom
    struct : structure object from Pymatgen
    sp : the mobile specie
    envtype : This sets the reference perfect structure. 'both' compares CSM_tet and CSM_oct and assigns to the lower one.
    'tet' refers to the perfect tetrahedron and 'oct' refers to the perfect octahedron
    result : a dictionary of environment information
    '''
    stripped = non_elements(struct, sp_anion)
    with_li = stripped.copy()
    with_li.append(sp, coord, coords_are_cartesian=False, validate_proximity=False)
    with_li = with_li.get_sorted_structure()  # (按照电负性排列) 将中心原子方法放到列表的第一个
    tet_oct_competition = []
    tet_neigh_coords_cart = []
    oct_neigh_coords_cart = []
    if envtype == 'both' or envtype == 'tet':
        # note 这个键长的阈值有待商榷 (可能会形成五棱锥)
        # [1.8, 2.1, 2.4, 2.7, 3, 3.3]
        # 确保第四根键和第五根键差值大于 0.3
        flag = True # 控制当len(neigh)==4时，再循环一次
        for dist in np.linspace(1.8, 3.3, 6):
            neigh = with_li.get_neighbors(with_li.sites[0], dist)
            if len(neigh) < 4:
                continue
            elif len(neigh) > 4:
                if len(tet_oct_competition) == 1:
                    tet_oct_competition.pop(0)
                break

            if len(neigh) == 4:
                if flag:
                    flag = False
                else:
                    break

            neigh_coords = [i.coords for i in neigh]
            tet_neigh_coords_cart = [periodicNeighbor.coords for periodicNeighbor in neigh]
            # 上面两个坐标是一样？
            with HiddenPrints():
                lgf = LocalGeometryFinder(only_symbols=["T:4"])
                lgf.setup_structure(structure=with_li)
                lgf.setup_local_geometry(isite=0, coords=neigh_coords)
            try:
                site_volume = ConvexHull(neigh_coords).volume
                tet_env_list = []
                # note 这里循环20次的原因是计算csm有一定的随机性？
                for i in range(2):
                    tet_env = {'csm': lgf.get_coordination_symmetry_measures()['T:4']['csm'],
                               'vol': site_volume,
                               'type': 'tet'}
                    tet_env_list.append(tet_env)
                tet_env = min(tet_env_list, key=lambda x: x['csm'])  # lambda指定了比较的键为 csm。
                tet_oct_competition.append(tet_env)

            except Exception as e:
                print(e)
                print("This site cannot be recognized as tetrahedral site")


    if envtype == 'both' or envtype == 'oct':
        # note 这个键长的阈值有待商榷 (可能会形成五棱锥)
        # [1.8, 2.1, 2.4, 2.7, 3, 3.3]
        flag = True  # 控制当len(neigh)==6时，再循环一次
        for dist in np.linspace(1.8, 3.3, 6):
            neigh = with_li.get_neighbors(with_li.sites[0], dist)
            if len(neigh) < 6:
                continue
            elif len(neigh) > 6:
                if len(tet_oct_competition) == 2:
                    tet_oct_competition.pop(1)
                break

            if len(neigh) == 6:
                if flag:
                    flag = False
                else:
                    break

            neigh_coords = [i.coords for i in neigh]
            oct_neigh_coords_cart = [periodicNeighbor.coords for periodicNeighbor in neigh]
            with HiddenPrints():
                lgf = LocalGeometryFinder(only_symbols=["O:6"], permutations_safe_override=False)
                lgf.setup_structure(structure=with_li)
                lgf.setup_local_geometry(isite=0, coords=neigh_coords)
            try:
                site_volume = ConvexHull(neigh_coords).volume
                oct_env_list = []
                for i in range(2):
                    oct_env = {'csm': lgf.get_coordination_symmetry_measures()['O:6']['csm'], 'vol': site_volume,
                               'type': 'oct'}
                    oct_env_list.append(oct_env)
                oct_env = min(oct_env_list, key=lambda x: x['csm'])
                tet_oct_competition.append(oct_env)

            except Exception as e:
                print(e)
                print("This site cannot be recognized as octahedral site")


    if len(tet_oct_competition) == 0:
        # return "None"
        return {'csm': np.nan, 'vol': np.nan, 'type': 'Non_' + envtype, 'distance_between_tet_centroid_and_tet_optimal_centroid': np.nan,
                                           'num_of_ligand': np.nan,
                                           'max_tet_optimal_centroid_angle_with_vertex': np.nan,'bond_std_dev': np.nan}

    elif len(tet_oct_competition) == 1:
        if tet_oct_competition[0]['type'] == 'tet':
            site_info_map = get_tet_oct_detail_info(tet_neigh_coords_cart, stripped, oct_coords_cart=None)

            # 'type' 'csm' 'vol'前面已经计算
            tet_oct_competition[0].update({'bond_std_dev': site_info_map['tet_bond_length_std_dev'],
                                           'distance_between_tet_centroid_and_tet_optimal_centroid': site_info_map['distance_between_tet_centroid_and_tet_optimal_centroid'],
                                           'max_tet_optimal_centroid_angle_with_vertex': site_info_map['max_tet_optimal_centroid_angle_with_vertex'],
                                           'num_of_ligand': site_info_map['num_of_ligand'],'tet_centroid':site_info_map['tet_centroid']})

        elif tet_oct_competition[0]['type'] == 'oct':
            # pass
            tet_oct_competition[0].update({'oct_neigh_coords_cart':oct_neigh_coords_cart})

        else:
            raise Exception(f"type格式错误！不是tet或者oct，为{tet_oct_competition[0]['type']}")

        return tet_oct_competition[0]

    elif len(tet_oct_competition) == 2:
        tet = tet_oct_competition[0]
        oct = tet_oct_competition[1]
        # oct
        if tet['csm'] > oct['csm']:
            oct.update({'oct_neigh_coords_cart':oct_neigh_coords_cart})
            return oct
        # tet
        else:
            site_info_map = get_tet_oct_detail_info(tet_neigh_coords_cart, stripped)

            # 'type' 'csm' 'vol'前面已经计算
            tet.update({'bond_std_dev': site_info_map['tet_bond_length_std_dev'],
                                           'distance_between_tet_centroid_and_tet_optimal_centroid': site_info_map[
                                               'distance_between_tet_centroid_and_tet_optimal_centroid'],
                                           'max_tet_optimal_centroid_angle_with_vertex': site_info_map[
                                               'max_tet_optimal_centroid_angle_with_vertex'],
                                           'num_of_ligand': site_info_map['num_of_ligand'],'tet_centroid':site_info_map['tet_centroid']})
            return tet


def divide_oct_2_4tet(oct_coords_list, structure, sp_anion):
    from mapping_sublattice.local_bcc_sublattice.find_all_bcc_tet_oct.find_all_bcc_site_v3 import get_optimal_4tet_in_oct_v2

    stripped = non_elements(structure, sp_anion)

    # 注意这里传入的要是阴离子结构
    tet_in_oct_info_list = get_optimal_4tet_in_oct_v2(oct_coords_list, stripped, tet_info_output_folder='temp', is_print=False, enable_logging=False)
    return tet_in_oct_info_list

def extract_sites(struct, sp="Li", sp_anion="Cl", envtype='both'):
    '''
    struct : structure object from Pymatgen
    envtype : 'tet', 'oct', or 'both'
    sp : target element to analyze environment
    '''
    envlist = []
    for i in range(len(struct.sites)):
        if struct.sites[i].specie != Element(sp):
            continue
        site = struct.sites[i]

        # 计算BV
        neigh = struct.get_neighbors(site, 4)
        bv = bond_valence.calculate_bv_sum(site, neigh, scale_factor=1.015)

        singleenv = site_env(site.frac_coords, struct, sp, sp_anion, envtype)
        if singleenv['type'] == 'tet':
            envlist.append({'frac_coords': site.frac_coords, 'type': singleenv['type'], 'csm': round(singleenv['csm'], 3),
                            'volume': round(singleenv['vol'], 3),
                            'distance_between_tet_centroid_and_tet_optimal_centroid': round(singleenv['distance_between_tet_centroid_and_tet_optimal_centroid'],3),
                            'max_tet_optimal_centroid_angle_with_vertex': round(
                                singleenv['max_tet_optimal_centroid_angle_with_vertex'], 3),
                            'bond_std_dev': round(
                                singleenv['bond_std_dev'], 3),
                            'num_of_ligand': singleenv['num_of_ligand'],
                            'bv': bv
                            })
        else:
            envlist.append(
                {'frac_coords': site.frac_coords, 'type': singleenv['type'], 'csm': round(singleenv['csm'], 3),
                 'volume': round(singleenv['vol'], 3), 'bv': bv})

    return envlist

def extract_sites_with_divided_tet(struct, sp="Li", sp_anion="Cl", envtype='both'):
    envlist = []
    for i in range(len(struct.sites)):
        if struct.sites[i].specie != Element(sp):
            continue
        site = struct.sites[i]

        # 计算BV
        neigh = struct.get_neighbors(site, 4)
        bv = bond_valence.calculate_bv_sum(site, neigh, scale_factor=1.015)

        singleenv = site_env(site.frac_coords, struct, sp, sp_anion, envtype)
        if singleenv['type'] == 'oct':
            tet_in_oct_info_list = divide_oct_2_4tet(singleenv['oct_neigh_coords_cart'], struct, sp_anion=sp_anion)

            envlist.append(
                {'frac_coords': site.frac_coords, 'type': singleenv['type'], 'csm': round(singleenv['csm'], 3),
                 'volume': round(singleenv['vol'], 3), 'bv': bv, 'tet_in_oct_info_list': tet_in_oct_info_list})

    return envlist

def process(structure: Structure, struc_name=None, site_info_filename = 'site_info.txt', sp = 'Li', sp_anion='Cl'):
    site_info = extract_sites(structure, sp=sp, sp_anion=sp_anion, envtype='both')

    with open(site_info_filename, 'a') as f:
        for item in site_info:
            if not struc_name:
                f.write(structure.reduced_formula + '\t' + item.get('type') + '\t' + str(item.get('csm')) + '\t' + str(item.get('volume')) + '\t'
                        + str(item.get('bond_std_dev')) + '\t'
                        + str(item.get('distance_between_tet_centroid_and_tet_optimal_centroid')) + '\t'
                        + str(item.get('max_tet_optimal_centroid_angle_with_vertex')) + '\t'
                        + str(item.get('num_of_ligand')) +'\t'+ str(item.get('bv')) +'\n')
            else:
                f.write(struc_name + '\t' + item.get('type') + '\t' + str(item.get('csm')) + '\t' + str(item.get('volume')) + '\t'
                        + str(item.get('bond_std_dev')) + '\t'
                        + str(item.get('distance_between_tet_centroid_and_tet_optimal_centroid')) + '\t'
                        + str(item.get('max_tet_optimal_centroid_angle_with_vertex')) + '\t'
                        + str(item.get('num_of_ligand')) +'\t'+ str(item.get('bv')) + '\n')
