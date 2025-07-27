from itertools import chain, combinations
import networkx as nx
import random
from sympy import symbols, And, Not
from sympy.logic.inference import satisfiable
import torch
from src.RBAMA import rescuing_net
from src.RBAMA import waiting_net
from src.environments import registered_versions
import numpy as np
import logging

class ReasoningUnit():
    def __init__(self, env, action_space):
        super().__init__()
        self.G = nx.DiGraph()
        self.action_space = action_space
        self.waiting_net =  waiting_net.On_Bridge(env)
        self.threshold_waiting = 0.8
        self.rescuing_net = rescuing_net.Rescuing(env)
        self.chosen_scenario = None
        self.static_background_info = set() #situationally independant rules; e.g. world knowledge (e.g. walking across the bridge -> arriving at the other side of the bridge); not used for the current implementation

        self.logger = logging.getLogger(__name__)

    @classmethod
    def int_to_subscript(self, number):
        subscript_offset = 8320  
        return ''.join(chr(subscript_offset + int(digit)) for digit in str(number))

    """
    updates the agent's reason theory based on feedback provided by a moral judge
    """
    def update(self, reason): #chosen (proper) scenario:the set of rules that the agent took for guiding its behavior

        #check if agent knows the morally relevant fact 
        if not reason[0] in self.G:
            self.G.add_node(reason[0], type='morally relevant fact')
        
        #check if action type is known to the agent as potential moral obligation
        if not reason[1] in self.G:
            self.G.add_node(reason[1], type='moral obligation')

        #check if connection between morally relevant fact and moral obligation is known to the agent
        if not self.G.has_edge(reason[0], reason[1]):
            edge_count = self.G.number_of_edges()
            self.G.add_edge(reason[0], reason[1], lower_order=set(), name=f"δ{self.int_to_subscript(edge_count+1)}")

        # correct the order such that the reason is prioritized over all reasons in the chosen scenario
        lower_order = self.G.get_edge_data(reason[0], reason[1])['lower_order']

        if self.chosen_scenario:
            for rule in self.chosen_scenario:
                if rule != reason:
                    lower_order.add(rule)
                
        nx.set_edge_attributes(self.G, {(reason[0], reason[1]): {'lower_order': lower_order}})
        self.log_edges_with_data()

    """
    log the agent's current reason theory
    """
    def log_edges_with_data(self):
        self.logger.info("Current reason theory:")
        for u, v, priority in self.G.edges(data=True):
            self.logger.info(f"Updated Edges: \nEdge from {u} to {v} with priority order: {priority}")


    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    """
    given the current state of the envrionment and the morally relevant facts, let the agent derive which moral obligation it has according to his current reason theory
    the proper scenarios are the ones for which the agent has no tiebreak; i.e. it chooses one randomly
    """
    def moral_obligations(self, lables, state):
        proper_scenarios = []
        moral_obligations = []
        morally_relevant_facts = lables
        for scenario in self.powerset(self.G.edges):
            if self.compute_binding(scenario, morally_relevant_facts, state) == set(scenario):
                proper_scenarios.append(set(scenario))
        #if there are several proper scenarios, let the agent choose one randomly; (buridan's ass)
        proper_scenario = random.choice(list(proper_scenarios))
        self.chosen_scenario = proper_scenario
        for rule in proper_scenario:
            moral_obligations.append(rule[1])
        return moral_obligations
    
    """
    compute the binding rules for a subset of the agent's rules
    """
    def compute_binding(self, scenario, morally_relevant_facts, state):
        background = symbols(morally_relevant_facts)
        antecedent = self.antecedent(background, scenario)
        triggered_rules = self.triggered(antecedent)
        conflicts = self.incompatibility_clauses(triggered_rules, state)
        background = background.union(conflicts)
        antecedent = self.antecedent(background, scenario)
        conflicted_rules = self.conflicted(antecedent)
        defeated_rules = self.defeated_rules(triggered_rules, background)

        binding_rules = {rule for rule in triggered_rules if rule not in conflicted_rules and rule not in defeated_rules}
        
        return binding_rules
    
    """
    given the set of conflicted abstract action types, construct a formula, that is used by the agent to derive its overall moral obligations
    """
    def incompatibility_clauses(self, triggered_rules, state):
        conclusions = set()
        for rule in triggered_rules:
            conclusions.add(rule[1])

        conflicted_action_sets = self.conflicted_actions(conclusions, state)

        formulas_conflicted = set()

        for conflicted_action_set in conflicted_action_sets:
            conflicting_action_types = symbols(conflicted_action_set) 
            formulas_conflicted.add(Not(And(*conflicting_action_types)))

        return formulas_conflicted
    
    """
    compute the set of moral oblgiations that are conflicting in a state (for which exists no primitive action is conform to all of them)
    """
    def conflicted_actions(self, action_types, state):

        conflicted = []

        combinations = self.powerset(action_types)
        for combination in combinations:
            morally_permissible = self.primitive_actions(set(combination),state)
            if not morally_permissible:
                conflicted.append(set(combination))

        return conflicted
    
    """
    compute the set of primitive actions that are conform with a moral obligation in a state
    """
    def primitive_actions(self, moral_obligations, state):
        action_space = [action for action in range(self.action_space)]
        morally_impermissible = []

        action_space = [action for action in range(self.action_space)]

        #if the agent has a moral obligation to rescue the drowning person, all actions but the preferred action of the rescue network are impermissible
        if 'R' in moral_obligations:
            moral_R = self.get_morally_optimal_action("R", state)
            morally_impermissible = [action for action in action_space if action != moral_R]

        #filters the action space to a subset that contains actions confrom with C
        if 'C' in moral_obligations:
            morally_impermissible.extend(self.get_morally_impermissible_actions('C', state))

        morally_permissible = [action for action in action_space if action not in morally_impermissible]

        return morally_permissible
    
    """returns the action that is optimal for rescuing a moral task (rescuing the drowning person)"""
    def get_morally_optimal_action(self, obligation, state):   
        if obligation == "R":
            dnn_input = self.rescuing_net.transformation(state)
            # Use the trained policy DQN to predict the Q-values for each action in the current state
            with torch.no_grad():
                q_values = self.rescuing_net.policy_dqn(dnn_input)
            moral_action = torch.argmax(q_values).item()
        return moral_action

    """filters out all actions that exceed a threshold and are therefore likely to violate a moral duty (not to push persons off the bridge)"""
    def get_morally_impermissible_actions(self, obligation, state):
        morally_impermissible_actions = []
        if obligation == "C":
            dnn_input = self.waiting_net.transformation(state)
            with torch.no_grad():
                q_values = self.waiting_net.policy_dqn(dnn_input)
            for q_value in q_values:
                if q_value >= self.threshold_waiting: 
                    actions_exceeding_theshold = torch.where(q_values == q_value)[0].tolist()
                    morally_impermissible_actions += actions_exceeding_theshold

        #if no action is sufficiently likely to be conform with the moral safety constraint, allow the one that is most likely to conform
        if len(morally_impermissible_actions) == len(range(self.action_space)):
            _, preferred_action = torch.max(q_values, dim=0)
            morally_impermissible_actions = [action for action in range(self.action_space) if action != preferred_action.item()]
        return morally_impermissible_actions
            
    
    """ 
    construct a conjunction of the background information and the conclusions of a subset of the the agent's default rules to be used as part of the agent's reasoning
    """
    def antecedent(self, background, scenario): 
        conclusions = set() 
        for rule in scenario:
            conc = rule[1]
            conclusions.add(symbols(conc))

        antecedent = conclusions.union(background).union(self.static_background_info)

        return And(*antecedent)

    def triggered(self, antecedent): #scnenario: a set of rules; true_propositions: set of labels

        triggered_rules= set()

        for rule in self.G.edges:
            prem = symbols(rule[0])
            if not satisfiable(And(antecedent, Not(prem))): 
                triggered_rules.add(rule)

        return triggered_rules
    
    def conflicted(self, antecedent): 

        conflicted_rules = set()

        for rule in self.G.edges:
            conc = symbols(rule[1])
            if not satisfiable(And(antecedent, conc)): 
                conflicted_rules.add(rule)

        return conflicted_rules
    
    def defeated_rules(self, triggered_rules, background):

        defeated_rules = set()

        for rule in self.G.edges: 
            for another_rule in triggered_rules:
                if rule in self.G.get_edge_data(another_rule[0], another_rule[1])['lower_order']:
                    to_add = symbols(another_rule[1])
                    to_add_set = set([to_add])
                    antecedent = background.union(to_add_set)
                    formla = And(*antecedent, symbols(rule[1]))
                    if not satisfiable(formla):
                        defeated_rules.add(rule)

        return defeated_rules
