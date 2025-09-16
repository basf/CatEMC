#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import NeighborList
from ase.data import covalent_radii


class RepulsivePotential(Calculator):
    """
    Repulsive potential defined as sum_ij U(xij), 
    defining xij = rij / [rc*(Ri + Rj)]
    where Ri and Rj are the radii of atom i an j, rij is their distance
    and rc a scaling constant, the potential U can take two forms:
    
    parabola form (goes to constant/prefactor at rij=0)
    
    U = prefactor * (xij-1)**exponent if rij<=xij
    U = 0 else

    Lennard jones like repulsive form (goes to infinity at rij=0)
    
    U = prefactor * [1/(xij**exponent)-1] - exponent*(1-xij)] if rij<=xij
    U = 0 else

    Examples
    --------
    >>> potential=RepulsivePotential(prefactor=..., rc=...)
    >>> to calculate, use as any ASE calculator, 
    e.g. potential.get_potential_energy()
    
    
    Parameters
    ----------
    prefactor: float, optional
        Strength constant for potentals
        Default is 1
    
    rc: float, optional
        Multplier on sum of atomic radii
        Default is 0.9
        (can also be imported as a list giving the radius of each atom)
    
    potential_type: str ('parabola' or 'LJ'), optional
        String specifying what mode of the potential is used. 
        Default is 'LJ' 
    
    exponent: float, optional
        Exponent used in potentials
        Default is 2
        
    extrapotentals: list of objects, optional
        List of additional objects capable of calculating, energy, 
        forces and stresses. Any objects with the following methods:
        potential(atoms),  forces(atoms),  stress(atoms)
        default: None (no extra potential used) 
    """

    implemented_properties = ['energy', 'forces', 'stress']

    default_parameters = {'prefactor': 1,'rc': 0.9, 
                          'potential_type': 'LJ', 'exponent': 2, 
                          'extrapotentials': None}    
    
    nolabel = True

    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'stress'],
                  system_changes=all_changes):
        """
        General ASE calculate method to store output in calculator.results
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # get parameters
        rc = self.get_radii(atoms)

        neighbor_list = NeighborList(np.array(rc), skin=0,
                                     self_interaction=False)     
        neighbor_list.update(atoms)

        energy, forces, stress = self.get_energy_and_derivatives(atoms, rc, neighbor_list)
        
        # set energy and forces
        self.results['energy'] = energy   
        self.results['forces'] = forces    
        
        assert(atoms.cell.rank==3)
        self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]] / atoms.get_volume()

        if self.parameters.extrapotentials is not None:
            assert isinstance(self.parameters.extrapotentials, list)
            self.get_extra_potential(atoms)


    def get_radii(self, atoms):
        """
        Method to get atomic radii.
        Overwritten in daughter class, don't Remove
        """
        
        if type(self.parameters.rc)==list:
            rc=self.parameters.rc
        else:
            covrads = [ covalent_radii[self.atoms[i].number] for i in range(len(self.atoms)) ]
            rc=self.parameters.rc*np.array(covrads)
    
        return rc



    def get_energy_and_derivatives(self, atoms, rc, neighbor_list):
        """
        Method to get energy, forces and stresses for the total atoms object
        using the repulsive prior function

        Parameters
        ----------
        atoms : ase.atoms
            Atoms under consideration
        rc : lits
            Atomic radii
        neighbors : ase.neighborlist
            Object containing all neighbor information

        Returns
        -------
        energy : floar
            Total energy
        forces : numpy.array
            Total forces
        stress : numpy.array
            Total stresses
        """
        
        energy, forces = self.setup_energy_forces(atoms)   
        
        stress = np.zeros((3, 3))

        for a1 in range(len(atoms)):
            
            neighbors, d = self.get_distance_info(atoms, a1, neighbor_list)
            
            if len(neighbors)>0:     
                
                potential, derivative = self.get_potential(a1, rc, d, neighbors)   
                
                energy = self.update_energy(energy, potential)
                
                forces = self.update_forces(forces, derivative, a1, neighbors) 
                
                stress=self.update_stress(stress, derivative, d)
                
        return energy, forces, stress
    
    
    
    def setup_energy_forces(self, atoms):
        """
        Method to setup output formats. 
        Overwritten in daughter classes. Dont remove
        """
        energy = 0.0
        forces = np.zeros((len(atoms), 3)) 
        return energy, forces



    def get_distance_info(self, atoms, atom_idx, neighbor_list):
        """
        Method to calculate the vectors between atoms in periodic as well
        as nonperiodic boundary conditions

        Overwritten in daughter classes. Dont Remove

        Parameters
        ----------
        atoms : ase.atoms
            Atoms object of consideration
        atom_idx : int
            index of one of the atoms in the atoms object
        neighbor_list : ase.neighborlist
            object containing all nerigborinformation

        Returns
        -------
        neighbors : list
            list of all neighbors of the given atom
        d : numpy.array
            vectors between all neighbors of the given atom
        """
        
        positions = atoms.positions
            
        cell = atoms.cell
        
        neighbors, offsets = neighbor_list.get_neighbors(atom_idx)
            
        cells = np.dot(offsets, cell)
            
        d = positions[neighbors] + cells - positions[atom_idx]
        
        return neighbors, d  
        

    def get_potential_function_LJ(self, prefactor, x, exp):
        """ 
        Method to calculate potential and derivative for the Lennard Jones 
        like repulsive potential
        """
        # Potential function with zero value and zero derivative at x=1.
        if x<=1:
            value = prefactor*(1/x**exp-1) - exp*prefactor*(1-x)
            deriv = exp*prefactor*(1-1/x**(exp+1))
        else:
            value = 0
            deriv = 0
        return value , deriv


    def get_potential_function_parabola(self, prefactor, x, exp):
        """ 
        Method to calculate potential and derivative for the parabolar 
        repulsive potential
        """
        # Potential function with zero value and zero derivative at x=1.
        
        if x<=1:
            value = prefactor*(x-1)**exp
            deriv = exp*prefactor*(x-1)**(exp-1)
        else:
            value = 0
            deriv = 0
        return value , deriv



    def get_x_potential(self, r, crs):
        """
        Get value and derivative of the repulsive potentials
        """
        
        x = r/crs
        
        potential = np.zeros(len(x))
        derivative=np.zeros(len(x))
        
        prefactor=self.parameters.prefactor
        exp=self.parameters.exponent
        
        if self.parameters.potential_type=='LJ':
            for i, xx in enumerate(x):
                potential[i], derivative[i] = self.get_potential_function_LJ(prefactor,xx, exp)
        elif self.parameters.potential_type=='parabola':
            for i, xx in enumerate(x):
                potential[i], derivative[i] = self.get_potential_function_parabola(prefactor,xx, exp)
        else:
            raise RuntimeError('potential_type={:s} not known.'
                               .format(self.parameters.potential_type))
            
        return potential, derivative 
        


    def radial_info(self, d, rc, atom_index, neighbors):
        """
        Get distances between atoms as well as the sum of atomic radii
        """
        
        r2 = (d**2).sum(1)
        r=np.sqrt(r2)
        
        crs=rc[atom_index]+np.array( [   rc[n] for n in neighbors   ]  )

        return r, crs


    def cartesian_conversion(self, derivative, d, r):
        """ Convert from polar to cartesian coordinates """
        r_block=np.tile(r,(len(d[0,:]),1)).T
        polar_to_cart=d/r_block
        derivative_cartesian=derivative[:,np.newaxis]*polar_to_cart
        
        return derivative_cartesian


    def get_potential(self, atom_index, rc, d, neighbors):     
        """
        Method to calculate the potential value and its derivatives
        in cartesian coordinates for a single atom and all its neighbors

        Parameters
        ----------
        atom_index : int
            Index of a given atom in the system
        rc : lits
            Atomic radii
        d : numpy.array
            Distance vectors between atom of index atom_index and all neighbors
        neighbors : ase.neighborlist
            Object containing all neighbor information

        Returns
        -------
        x_potential : float
            the energy
        cartesian_derivatives: numpy.array
            the derivatives of the energy
        """
        r, crs = self.radial_info(d, rc, atom_index, neighbors)  

        x_potential, x_potential_derivative = self.get_x_potential(r, crs)

        derivative = x_potential_derivative * 1/crs

        cartesian_derivative = self.cartesian_conversion(derivative, d, r)  

        return x_potential, -cartesian_derivative

    
    def update_energy(self, energy, potential):
        """ Method to update the total energy """
        energy += potential.sum()
        return energy
    
    
    def update_forces(self, forces, derivative, atom_index, neighbors):
        """ Method to update the total forces """
        forces[atom_index] -= derivative.sum(axis=0)
    
        for a2, f2 in zip(neighbors, derivative):
            forces[a2] += f2
        return forces
    
    def update_stress(self, stress, derivative, d):
        """ Method to update the total stresses """
        stress -= np.dot(derivative.T, d)
        return stress

    def get_extra_potential(self, atoms):
        """
        Method to add extrapotential values on top of the repulsive potential
        """
        for potential in self.parameters.extrapotentials:  
            self.results['energy']  += potential.potential(atoms)
            self.results['forces'] += potential.forces(atoms)
            self.results['stress']  +=  potential.stress(atoms)
