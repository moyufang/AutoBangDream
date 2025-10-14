from enum import Enum, auto
from configuration import Event

MAX_TARGET_PT = 3000
MAX_HIGH_SCORE = 3000000
MIN_SCORE_RATE = 0.01

class ErrorID(Enum):
	TargetPtTooLarge = auto()
	HighScoreTooLarge = auto()
	NegativeValue = auto()

class Error(ValueError):
	def __init__(self, error_id:ErrorID, atached_hint:str=None):
		if (error_id == ErrorID.TargetPtTooLarge):
			error_hint = f"Inputed target pt exceeded the limited: {MAX_TARGET_PT}."
		elif (error_id == ErrorID.HighScoreTooLarge):
			error_hint = f'Team with propertities ({atached_hint}) has an banned high_score exceeded: {MAX_HIGH_SCORE}.'
		elif (error_id == ErrorID.NegativeValue):
			error_hint = f'Negative Value: {atached_hint}'
		super().__init__(self, error_hint);

class Team:
	def __init__(self, addition:int, high_score:int, shift_pt:int=0):
		if high_score > MAX_HIGH_SCORE: raise Error(ErrorID.HighScoreTooLarge,
			f"{addition},{high_score},{shift_pt}")
		self.addition = addition
		self.high_score = high_score
		self.shift_pt = shift_pt

class Event(Enum):
	Mission = auto()
	Trial = auto()
	Challenge = auto()
	Compete = auto()
	Team = auto()
	Tour = auto()

def pt_to_score(event:Event, target_pt:int, team_list:list):
	if (target_pt > MAX_TARGET_PT): raise Error(ErrorID.TargetPtTooLarge)

	if (event == Event.Mission):
		base_pt = 120; score_factor = 15000
	elif (event == Event.Trial):
		base_pt = 130; score_factor = 26000
	elif (event == Event.Challenge):
		base_pt = 70; score_factor = 50000
	elif (event == Event.Compete):
		base_pt = 123; score_factor = 6500
	elif (event == Event.Team):
		base_pt = 0+105+50; score_factor = 6500
	elif (event == Event.Tour):
		base_pt = 30; score_factor = 18500

	if event in [Event.Mission, Event.Trial, Event.Challenge]:
		mp = {}
		f = [-1]*(target_pt+1); f[0] = 0
		for i,team in enumerate(team_list):
			max_s = team.high_score//score_factor
			for s in range(1, max_s+1):
				if s/max_s < MIN_SCORE_RATE: continue
				pt = ((base_pt+s)*(100+team.addition))//100+team.shift_pt
				ori_s = ((pt-team.shift_pt)*100+99+team.addition)//(100+team.addition)-base_pt
				assert ori_s == s
				if pt in mp: continue
				mp[pt] = i
		for pt in range(0, target_pt):
			if f[pt] == -1: continue
			for delta_pt in mp:
				if 	delta_pt+pt <= target_pt and\
					(f[delta_pt+pt] == -1 or f[delta_pt+pt] > f[pt]+1):
					f[delta_pt+pt] = f[pt]+1
		if f[target_pt] == -1:
			return None
		min_step = f[target_pt]; avg_pt = target_pt/min_step;
		g = [-1]*(target_pt+1)
		for pt in range(0, target_pt):
			if f[pt] == -1: continue
			for delta_pt in mp:
				if delta_pt+pt > target_pt or f[delta_pt+pt] == -1 or f[delta_pt+pt] != f[pt]+1: continue
				if g[delta_pt+pt] == -1 or \
					abs(g[delta_pt+pt]-avg_pt)>abs(delta_pt-avg_pt):
					g[delta_pt+pt] = delta_pt
		pt_path = []
		tpt = target_pt
		for i in range(min_step):
			if g[tpt] == -1: raise AssertionError("Path False.")
			delta_pt = g[tpt]
			team_id = mp[delta_pt]
			team = team_list[team_id]
			s = ((delta_pt-team.shift_pt)*100+99+team.addition)//(100+team.addition)-base_pt
			pt_path.append([team_id, delta_pt, score_factor*s, score_factor*(s+1)-1])
			tpt = tpt-g[tpt]
		if tpt != 0: raise AssertionError("Path False with no targed pt.")
		return pt_path
	else:
		return [(target_pt-base_pt)*score_factor, (target_pt-base_pt)*(score_factor+1)-1]

"""
本控分器的基本原理是:
  你提供 "pt 数"、"主乐队综合力" 和 "活动加成" (任务活动还需提供 "副队综合力") 
  控分器给出 "要演出多少次 (一般1、2次) "、"每次演出要打到的分数区间"
  你在挂机号的辅助下打进对应分数区间, 获得所需pt

对控分来说, 活动分两类: 组曲Live 和 非组曲Live
这是因为除了组曲Live之外的活动, 通过开房都能做到在0血时演出继续, 
	如果这时候所有人都不再打note的话, 分数就会一直保持到演出结束, 给控分带来极大方便。
  但要注意邦邦演出的惩罚机制, 如果某次演出中miss数大于该曲子总note数的一半, 
  就会被认为在破坏 BangDream 协力环境，然后触发惩罚机制, 清 0 本次演出的 pt, 且不退还消耗的火。

非组曲活控分方法

使用本控分计算器的基本流程是
 (1) 任选一个或一些配队, 提供相关参数给本控分计算器, 得到控分路径
	一个控分路径包含若干次演出, 每次演出给出 使用哪个配队、本次演出要打的pt、能达成本次pt的分数区间
 (2) 为完成路径上的某一次演出, 开一个由控分号、辅助号组成的两人房间, 控分号必须使用0火
 (3) 演出开始后, 辅助号挂机并保持永远0分
 (4) 控分号的操作是打到对应分数区间，整个过程辅助号一直放置挂机
	分两种情况
		一、分数区间的值较大
			这时候控分手在演出时应时刻注意左上角的分数, 
			一旦达到分数区间就立即放手, 等待演出结束。
			如果不小心打多了使得左上角的分数超出了分数区间, 则本次演出控分失败, 
			请立即切屏退出此次演出, 只要没有进入结算界面, 就还有再来的机会。
			因为分数区间的值较大, note打的较多, 打到目标区间就放手并不会触发惩罚机制, 能成功获得pt
		二、分数区间的值较小
			这时候控分手可以先打到里分数区间接近的分数, 然后悬空双手放血, 
			0血后, 得分是未空血时的十分之一, 
			可以趁此敲击大量的note以达到目标分数区间, 且保证演出结束后miss数不超过一半
 (5) 控分号在不触发惩罚机制且成功打到目标分数区间后, 该次演出成功, 
	可以通过结算画面中看到的获得的pt数来验证控分路径的正确性。
	完整正确打完控分路径上的每一次演出, 即为控分成功。

了解了基本流程, 就可以谈谈该给本控分计算器的相关特性了
 (1) 每期活动, 不同乐队有不同加成, 对于一个特定的加成, 并不是所有目标pt都能打到
		比如这种情况 配队加成为200%, 本次活动的基础pt为120, 副队pt为90, 
			那么无论如何每次演出能打出pt数必定是3的倍数, 
			只要目标pt不是3的倍数, 只用该配队的控分路径是不存在的。
			但注意到如果再提供一个有着可以打出非3倍数pt的加成的配队, 
			利用两个配队就必定能对任何目标pt数找到合法的控分路径。
		所以提供的乐队的加成种类越多, 有解的目标pt数就更多, 
		所以本控分计算器允许同时输入多个加成不同的乐队, 这样有解的概率更大, 
		实际使用中, 对某目标pt往往只需要提供一个或两个乐队, 就能成功找到解
 (2) 不同活动中, pt 计算公式不同，控分计算器也有小小差异

以下计算公式中, 中括号 [] 代表向下取整

1. Event.Mission		#任务Live
一次0火队友0分的协力演出的pt计算公式: 
pt=[(120+[得分/15000])*加成倍率]+副队pt

任务Live中涉及到副队pt, 需要提供副队pt, 
可以在游戏中“乐队组成”ui界面中找到查看“副队综合力”的按钮, 副队pt=[副队综合力/3000] (向下取整) 
其它活动中没有副队, 这一项输入为0即可 (必须为0, 否则计算出错) 

2. Event.Trial			#试炼Live(EX牌活)
一次0火队友0分的协力演出的pt计算公式: 
pt=[(130+[得分/26000])*加成倍率]

除了没有副队, 其它与任务Live相同。
注意, 试炼Live中, 完成试炼完成是会给奖励pt的, 请保证控分时不会触发试炼任务的完成

3. Event.Challenge		#挑战Live
一次0火队友0分的协力演出的pt计算公式: 
pt=[(70+[得分/50000])*加成倍率]

控分就不要用cp啦, 同样用协力来解决。

4. Event.Compete 		#竞演Live
一次0火队友0分的双人竞演演出的pt计算公式: 
pt=(123+[得分/6500])
因为队友 0 分的缘故且只有的两人的缘故, 控分号结算时必定rk2, 也即获得123pt的基础pt

5. Event.Team	#团竞Live
一次0火队友0分的双人2v5演出的pt计算公式: 
pt=(105+50+[得分/6500])
默认对方团队会获胜、队友 0 分且只有的两人, 获得155pt的基础pt
如果真的出现对方团队失败的 (几乎不可能发生) 的情况, 那么控分失败, 请及时切屏退出。

组曲活控分方法

6. Event.Tour		#组曲Live
一把0火且进行一次演出的pt计算公式: 
pt = 30+[得分/18500]

组曲活算是最难控分的活动了, 因为协力不加pt, 
且只能在巡回演出中获得打满至少一首的pt, 
	因为如果第二把中途死亡, 分数会降会到原先只打完第一首时的分数。
注意到加成并不影响pt, 需要控分手自己去组能够打到目标分数区间的配队和歌曲, 
亲自尝试容易发现, 这并不困难。
比较容易的配队和选曲方法是选择综合力较低的配队、打easy, 多打几首会更容易控分。

总结
实际上还可以把活动这样分成两类: 
 (1) 任务Live、试炼Live、挑战Live算一类, 因为加成影响pt
 (2) 竞演Live、团竞Live、组曲Live算一类, 因为加成只影响综合力, 不影响pt

"""

"""
Event.Mission		  #任务Live
Event.Trial			  #试炼Live(EX牌活)
Event.Challenge		#挑战Live
Event.Compete 		#竞演Live
Event.Team		    #团队竞演Live
Event.Tour			  #组曲Live
"""

pt_path = pt_to_score(
	event=Event.Trial, #活动类型
	target_pt=950,
	team_list=[	
		#配队加成%、全漏技能键的单次演出能打到的最高分(近似即可)、副队pt(副队综合力 // 3000)
		#Team(350, 1600000, 85),
		#Team(100, 1000000, 296742//3000)
		#Team(300, 1000000, 85)
		Team(252, 1800000, 0),
	]
)
if pt_path == None:
	print("控分路径不存在")
	exit(0)
for o in pt_path:
	print(o)