# -*- coding: utf-8 -*-
"""
This is a tokenizer for programming-specific or software-specific texts in Stack Overflow.
Such text is both social, e.g., people use ungrammatical and informal language in comments,
and domain-specific, e.g., printf() should be recognized as one token rather than 3 tokens printf, '(' and ')'.

Acknowledgement goes to Brendan O'Connor, Kevin Gimpel and Daniel Mills,
who are the authors of a Twitter tokenizer. This tokenizer modifies their Twitter one.


general philosophy is to throw as little out as possible.
development philosophy: every time you change a rule, do a diff of this
program's output on ~100k tweets.  if you iterate through many possible rules
and only accept the ones that seeem to result in good diffs, it's a sort of
statistical learning with in-the-loop human evaluation :)
"""


import re,sys
#import embeddings.emoticons as emoticons

""" emoticon recognition via patterns.  tested on english-language twitter, but
probably works for other social media dialects. """

__author__ = "Brendan O'Connor (brenocon.com, brenocon@gmail.com)"
"Modified for part-of-speech tagging by Kevin Gimpel (kgimpel@cs.cmu.edu) and Daniel Mills (dpmills@cs.cmu.edu)"

#from __future__ import print_function
import re,sys

mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
#SMILEY = mycompile(r'[:=].{0,1}[\)dpD]')
#MULTITOK_SMILEY = mycompile(r' : [\)dp]')

NormalEyes = r'[:=8]'
Wink = r'[;]'

NoseArea = r'(|o|O|-)'   ## rather tight precision, \S might be reasonable...

HappyMouths = r'[D\)\]]'
SadMouths = r'[\(\[]'
Tongue = r'[pPb]'
OtherMouths = r'[\|doO/\\]'  # remove forward slash if http://'s aren't cleaned

Happy_RE =  mycompile( '(\^_\^|' + NormalEyes + NoseArea + HappyMouths + ')')
Sad_RE = mycompile(NormalEyes + NoseArea + SadMouths)

Wink_RE = mycompile(Wink + NoseArea + HappyMouths)
Tongue_RE = mycompile(NormalEyes + NoseArea + Tongue)
Other_RE = mycompile( '('+NormalEyes+'|'+Wink+')'  + NoseArea + OtherMouths )


Faces = (
    "(" +
    "("+NormalEyes+"|"+Wink+")" +
    NoseArea +
    "("+Tongue+"|"+OtherMouths+"|"+SadMouths+"+|"+HappyMouths+"+)" +
    "|" +
    "("+SadMouths+"+|"+HappyMouths+"+)" +
    NoseArea +
    "("+NormalEyes+"|"+Wink+")" +
    ")"
)

Hearts = r'(<+/?3+)'

Arrows = r'(<*[-=]*>+|<+[-=]*>*)'

Emoticon = "("+Hearts+"|"+Faces+"|"+Arrows+")"

Emoticon_RE = mycompile(Emoticon)

#Emoticon_RE = "|".join([Happy_RE,Sad_RE,Wink_RE,Tongue_RE,Other_RE])
#Emoticon_RE = mycompile(Emoticon_RE)

def analyze_tweet(text):
  h= Happy_RE.search(text)
  s= Sad_RE.search(text)
  if h and s: return "BOTH_HS"
  if h: return "HAPPY"
  if s: return "SAD"
  return "NA"

  # more complex & harder, so disabled for now
  #w= Wink_RE.search(text)
  #t= Tongue_RE.search(text)
  #a= Other_RE.search(text)
  #h,w,s,t,a = [bool(x) for x in [h,w,s,t,a]]
  #if sum([h,w,s,t,a])>1: return "MULTIPLE"
  #if sum([h,w,s,t,a])==1:
  #  if h: return "HAPPY"
  #  if s: return "SAD"
  #  if w: return "WINK"
  #  if a: return "OTHER"
  #  if t: return "TONGUE"
  #return "NA"

#import util
mycompile = lambda pat:  re.compile(pat,  re.UNICODE)
def regex_or(*items):
  r = '|'.join(items)
  r = '(' + r + ')'
  return r
def pos_lookahead(r):
  return '(?=' + r + ')'
def neg_lookahead(r):
  return '(?!' + r + ')'
def optional(r):
  return '(%s)?' % r

PunctChars = r'''['“".?!,(/:;]'''
foo = r'''foo'''
Punct = '%s+' % PunctChars
PunctSeq = r"""['`\"“”‘’)]+|[.?!,…]+|[:;/(]+"""
Entity = '&(amp|lt|gt|quot);'

# one-liner URL recognition:
#Url = r'''https?://\S+'''

# more complex version:
UrlStart1 = regex_or('https?://', r'www\.')
CommonTLDs = regex_or('com','co\\.uk','org','net','info','ca','edu','gov')
UrlStart2 = r'[a-z0-9\.-]+?' + r'\.' + CommonTLDs + pos_lookahead(r'[/ \W\b]')
UrlBody = r'[^ \t\r\n<>]*?'  # * not + for case of:  "go to bla.com." -- don't want period
UrlExtraCrapBeforeEnd = '%s+?' % regex_or(PunctChars, Entity)
UrlEnd = regex_or( r'\.\.+', r'[<>]', r'\s', '$') # / added by Deheng

Url = (r'\b' +
    regex_or(UrlStart1, UrlStart2) +
    UrlBody +
    pos_lookahead( optional(UrlExtraCrapBeforeEnd) + UrlEnd))

Url_RE = re.compile("(%s)" % Url, re.U|re.I)

EmailName = r'([a-zA-Z0-9-_+]+[.])*[a-zA-Z0-9-_+]+'
EmailDomain = r'([a-zA-Z0-9]+[.])+' + CommonTLDs
Email = EmailName + "@" + EmailDomain

Email_RE = mycompile(Email)

API = regex_or(r'((\w+)\.)+\w+\(\)', r'\w+\(\)', r'((\w+)\.)+(\w+)')
API_RE = mycompile(API)
plural = r'\w+\(s\)'
plural_RE = mycompile(plural)
ProgrammingOperators = regex_or(r'==', r'!=', r'>=', r'<=', r'&&', r'\|\|')

Concat = r'\w+[/—]\w+'

Timelike = r'\d+:\d+'
NumNum = r'\d+\.\d+'
NumberWithCommas = r'(\d+,)+?\d{3}' + pos_lookahead(regex_or('[^,]','$'))

Abbrevs1 = ['am','pm','us','usa','ie','eg']
def regexify_abbrev(a):
  chars = list(a)
  icase = ["[%s%s]" % (c,c.upper()) for c in chars]
  dotted = [r'%s\.' % x for x in icase]
  return "".join(dotted)
Abbrevs = [regexify_abbrev(a) for a in Abbrevs1]

BoundaryNotDot = regex_or(r'\s', '[“"?!,:;]', Entity)
aa1 = r'''([A-Za-z]\.){2,}''' + pos_lookahead(BoundaryNotDot)
aa2 = r'''([A-Za-z]\.){1,}[A-Za-z]''' + pos_lookahead(BoundaryNotDot)
ArbitraryAbbrev = regex_or(aa1,aa2)

assert '-' != '―'
Separators = regex_or('--+', '―')
Decorations = r' [  ♫   ]+ '.replace(' ','')

EmbeddedApostrophe = r"\S+'\w+"  # \S -> \w, by Deheng

Hashtag = r'#[a-zA-Z0-9_]+'

AtMention = r'@[a-zA-Z0-9_]+'

ProtectThese = [
    Emoticon,
    Url,
    Email,
    Entity,
    Hashtag,
    AtMention,
    Timelike,
    NumNum,
    NumberWithCommas,
    ArbitraryAbbrev,
    Separators,
    Decorations,
    EmbeddedApostrophe,
    API,  # Deheng
    PunctSeq,  # Deheng
    plural,   # Deheng
    ProgrammingOperators  # Deheng
]
Protect_RE = mycompile(regex_or(*ProtectThese))


class Tokenization(list):
  " list of tokens, plus extra info "
  def __init__(self):
    self.alignments = []
    self.text = ""
  def subset(self, tok_inds):
    new = Tokenization()
    new += [self[i] for i in tok_inds]
    new.alignments = [self.alignments[i] for i in tok_inds]
    new.text = self.text
    return new
  def assert_consistent(t):
    assert len(t) == len(t.alignments)
    assert [t.text[t.alignments[i] : (t.alignments[i]+len(t[i]))] for i in range(len(t))] == list(t)

def align(toks, orig):
  s_i = 0
  alignments = [None]*len(toks)
  for tok_i in range(len(toks)):
    while True:
      L = len(toks[tok_i])
      if orig[s_i:(s_i+L)] == toks[tok_i]:
        alignments[tok_i] = s_i
        s_i += L
        break
      s_i += 1
      if s_i >= len(orig): raise AlignmentFailed((orig,toks,alignments))
      #if orig[s_i] != ' ': raise AlignmentFailed("nonspace advance: %s" % ((s_i,orig),))
  if any(a is None for a in alignments): raise AlignmentFailed((orig,toks,alignments))

  return alignments

class AlignmentFailed(Exception): pass

def unicodify(s, encoding='utf8', *args):
  if isinstance(s,str): return s
  if isinstance(s,str): return s.decode(encoding, *args)
  return unicode(s)

def tokenize(tweet):
  #tweet = re.sub(u'—', ' ', tweet) # Deheng
  #tweet = tweet.replace('\x97', ' ') # Deheng
  text = unicodify(tweet)
  text = re.sub(u'—', ' ', text) # Deheng
  text = re.sub('(?<=\w)/(\s|$)', ' ', text) # Deheng
  text = squeeze_whitespace(text)
  # Convert HTML escape sequences into their actual characters (so that the tokenizer and emoticon finder are not fooled).
  text = re.sub(r'&lt;', '<', text)
  text = re.sub(r'&gt;', '>', text)
  text = re.sub(r'&amp;', '&', text)

  t = Tokenization()
  t += simple_tokenize(text)
  t.text = text
  t.alignments = align(t, text)
  return t

def simple_tokenize(text):
  s = text
  s = edge_punct_munge(s)

  # strict alternating ordering through the string.  first and last are goods.
  # good bad good bad good bad good
  goods = []
  bads = []
  i = 0
  if Protect_RE.search(s):
    for m in Protect_RE.finditer(s):
      goods.append( (i, m.start()) )
      bads.append(m.span())
      i = m.end()
    goods.append( (m.end(), len(s)) )
  else:
    goods = [ (0, len(s)) ]
  assert len(bads)+1 == len(goods)

  goods = [s[i:j] for i,j in goods]
  bads  = [s[i:j] for i,j in bads]
  # print goods
  # print bads
  goods = [unprotected_tokenize(x) for x in goods]
  res = []
  for i in range(len(bads)):
    res += goods[i]
    res.append(bads[i])
  res += goods[-1]
  #res = post_process(res)
  return res

AposS = mycompile(r"(\S+)('s)$")

def post_process(pre_toks):
  # hacky: further splitting of certain tokens
  post_toks = []
  for tok in pre_toks:
    m = AposS.search(tok)
    if m:
      post_toks += m.groups()
    else:
      post_toks.append( tok )
  return post_toks

WS_RE = mycompile(r'\s+')
def squeeze_whitespace(s):
  new_string = WS_RE.sub(" ",s)
  return new_string.strip()

# fun: copy and paste outta http://en.wikipedia.org/wiki/Smart_quotes
#EdgePunct      = r"""[  ' " “ ” ‘ ’ * < > « » { } ( \) [ \]  ]""".replace(' ','')
# Removed < and > because they were causing problems with <3 emoticons
EdgePunct      = r"""[  ' " “ ” ‘ ’ * « » { } ( \) [ \]  ]""".replace(' ','')
#EdgePunct = r'[^a-zA-Z0-9\s]'
#NotEdgePunct = r"""[^'"([\)\]]"""  # alignment failures?
NotEdgePunct = r"""[a-zA-Z0-9]"""
EdgePunctLeft  = r"""(\s|^)(%s+)(%s)""" % (EdgePunct, NotEdgePunct)

# programming API suffixes () are considered
EdgePunctRight =   r"""(%s(?:\(s?\))?)(%s*)(\s|$)""" % (NotEdgePunct, EdgePunct)
EdgePunctLeft_RE = mycompile(EdgePunctLeft)
EdgePunctRight_RE= mycompile(EdgePunctRight)

def edge_punct_munge(s):
  s = EdgePunctLeft_RE.sub( r"\1\2 \3", s)
  #print s
  s = EdgePunctRight_RE.sub(r"\1 \2\3", s)
  #print s
  return s


def unprotected_tokenize(s):
  return s.split()

if __name__=='__main__':
  # for line in sys.stdin:
  #   print(" ".join(tokenize(line[:-1])).encode('utf-8'))

  text = """I replace self.conv1() = nn.Conv2d(1, 32, 5, padding=2) with self.w_conv1 = Variable(torch.randn(1, 32, 5))"""
  print(tokenize(text))
  print(" ".join(tokenize(text)).encode('utf-8'))
  #print "CUR\t" + " ".join(tokenize(line[:-1]))
    #print "WS\t" + " ".join(line[:-1].split())
    #print ansi.color(line.strip(),'red')
    #print ansi.color(" ".join(tokenize(line.strip())),'blue','bold')

