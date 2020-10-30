"""
Authors:
 - Ayushman Dash <ayushman@neuralspace.ai>
"""
from panini.constants import POS
from panini.nlp import PaniniNLP
from panini.types.generic import ModelCollection

nlp = PaniniNLP(model=ModelCollection.hindi)
doc = nlp(
    "बुधवार को संजय दत्त (Sanjay Dutt) को एक लोकप्रिय सैलून के बाहर देखा गया था. "
    "वह यहां अपने बाल कटवाने के लिए पहुंचे थे. "
    "संजय लोकप्रिय हेयर स्टाइलिस्ट आलिम हाकिम के सैलून गए थे. "
    "इस चर्चित हेयर स्टाइलिस्ट ने अपने इंस्टाग्राम पेज पर जाकर अभिनेता के दो वीडियो शेयर किए हैं."
)


for tok in doc.tokens:
    print(f"{tok.text}\t{tok.get_tag(POS)}")
