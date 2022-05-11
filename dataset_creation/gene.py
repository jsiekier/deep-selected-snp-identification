
class Gene:
    def __init__(self,seqid,source,feature,start,end,score,strand,frame,attributes,gene_id):
        self.seqid=seqid
        self.source=source
        self.feature=feature
        self.start=int(start)-1
        self.end=int(end)
        self.score=score
        self.strand=strand
        self.frame=frame
        self.gene_id=gene_id
        if type(attributes) is str:
            self.attributes=dict()
            attributes=attributes.replace("\n","")
            for attr in attributes.split(";"):
                key,val=attr.split("=")
                self.attributes[key]=val
        else:
            self.attributes=attributes
        #TODO: init with numpy arr
        self.alingment=[]
        self.position=[]
        self.base=[]

