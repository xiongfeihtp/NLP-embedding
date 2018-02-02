from bosonnlp import BosonNLP


token = 'Qa3gttjU.16601.0IxIZCYg0CIa'
nlp = BosonNLP(token)


def get_summary_from_boson(content, title):
    fit_length = 200
    limit_ratio = fit_length / len(content)
    summary = nlp.summary(title, content, word_limit=limit_ratio)
    return summary

if __name__ == '__main__':
    content = """中青在线华盛顿7月12日电（中国青年报•中青在线驻美国记者 刘平）持续发酵的美国“通俄门”爆出最新猛料。7月11日，备受压力的特朗普长子公布了他与一名俄罗斯律师的往来电邮，而这位律师是一位“中间人”，据称掌握有关希拉里的负面消息。
“猛料”一出，众议院少数党领袖佩洛西呼吁，包括小特朗普在内的所有相关人士必须立即前往国会宣誓并听证。美国媒体则报道称，白宫现在已陷入一片混乱。
小特朗普公布的电子邮件显示，2016年6月3日，一个名为罗布•戈德斯通的人给他发来电邮：“艾敏刚给我打电话，要我就一些非常有意义的事联系你。俄罗斯联邦总检察长愿意提供一些官方文件和信息给特朗普竞选团队。这些东西与希拉里有关，她与俄罗斯之间的往来记录会对你的父亲非常有用。这些信息的来源非常高层，且极为敏感，是俄罗斯及其政府对特朗普先生的支持的一部分。阿拉斯和艾敏在居中牵线。你认为处理这些信息的最好方式是什么?你需要和艾敏直接讨论这事吗?我也可以通过罗娜把这些信息发给你父亲，但因为这极其敏感，所以想先发给你。”
收到邮件十几分钟后，小特朗普回邮件说：“非常感谢。我现在在路上，也许会先直接和艾敏联系。看起来还有一些时间——如果这就是你说的我非常喜欢的东西，尤其是在今年夏季末期。”
发邮件的罗布∙戈德斯通是一名公关人员，当时是俄罗斯歌星艾敏•阿加拉罗夫的经纪人，艾敏•阿加拉罗夫则是有“俄罗斯的特朗普”之称的房地产大亨阿拉斯·•阿加拉罗夫之子。2013年，阿拉斯将当时由特朗普经营的“环球小姐”选美活动引入俄罗斯，同年特朗普本人曾出现在艾敏的音乐专辑中，贡献了其电视真人秀节目“学徒”中的那句有名的“你被解雇了”的台词。其时，因为共和党内最后一名对手克鲁兹在5月3日宣布退选，特朗普已稳获该党总统候选人提名。
2016年6月6日，小特朗普与艾敏通了电话，随后通过邮件对戈德斯通表示了感谢。戈德斯通很快在邮件中表示，“俄罗斯政府的律师将从莫斯科飞过去”与小特朗普见面。6月8日，小特朗普给他的妹夫库什纳和时任特朗普竞选团队主席马纳福特发邮件，通知他们此次会面安排在次日下午4时，地点在他的办公室。
《纽约时报》今年7月9日率先报道了小特朗普等人与那位俄罗斯律师维塞尼茨卡雅会面的有关情况，并告知小特朗普，该报将在近期刊出披露有关电邮的报道。在此背景下，小特朗普公布了这些电邮。
特朗普总统表示支持其儿子。7月11日，白宫副发言人莎拉•桑德斯宣读了特朗普总统的声明： “我儿子是一个高素质的人，我赞赏他的透明度。”桑德斯表示，总统对“通俄门”事件持续发酵感到沮丧，他迫切希望能够专注于税改和基础设施建设等其他事情。
在小特朗普公开这些电邮前，白宫曾为小特朗普辩护，称其与维塞尼茨卡雅会面没有包含任何（不恰到的）东西，这恰恰证明了特朗普竞选团队与俄罗斯之间没有任何旨在帮助特朗普的“勾结”。
美国参议院多数党领袖麦康奈尔7月11日表示，“他们会追查到底的”。《华盛顿邮报》引多名白宫高官及外围高级顾问的话称，这是一场“5级飓风”和现实版“纸牌屋”，白宫已陷入一片混乱。（国际部编辑）
责编：李伊涵"""
    title = """“通俄门”爆出最新猛料—— 特朗普长子公开邮件令白宫陷入混乱"""
    summary = get_summary_from_boson(content, title)
    print(summary)

