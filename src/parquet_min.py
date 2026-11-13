from pathlib import Path
import ctypes
import io
import struct
from thrift.transport.TTransport import TMemoryBuffer
from thrift.protocol.TCompactProtocol import TCompactProtocol
from thrift.Thrift import TType
import numpy as np

# Thrift compact encoded Parquet metadata (read-only, column primitives)
def parse_any(p, t):
    if t == TType.STRUCT:
        return parse_struct(p)
    if t == TType.LIST:
        t2, n = p.readListBegin()
        out = [parse_any(p, t2) for _ in range(n)]
        p.readListEnd()
        return out
    if t == TType.SET:
        t2,n=p.readSetBegin()
        out=[parse_any(p,t2) for _ in range(n)]
        p.readSetEnd()
        return out
    if t == TType.MAP:
        kt,vt,n=p.readMapBegin()
        out={parse_any(p,kt):parse_any(p,vt) for _ in range(n)}
        p.readMapEnd()
        return out
    if t == TType.BOOL:return p.readBool()
    if t == TType.BYTE:return p.readByte()
    if t == TType.I16:return p.readI16()
    if t == TType.I32:return p.readI32()
    if t == TType.I64:return p.readI64()
    if t == TType.DOUBLE:return p.readDouble()
    if t == TType.STRING:return p.readBinary()
    raise ValueError('ttype:'+str(t))

def parse_struct(p):
    p.readStructBegin()
    d={}
    while True:
        nm,t,fid=p.readFieldBegin()
        if t==TType.STOP: break
        d[fid]=parse_any(p,t)
        p.readFieldEnd()
    p.readStructEnd()
    return d

def read_meta(path):
    with open(path,'rb') as fp:
        fp.seek(-8,2)
        mlen=struct.unpack('<I', fp.read(4))[0]
        assert fp.read(4)==b'PAR1'
        fp.seek(-8-mlen,2)
        data=fp.read(mlen)
    return parse_struct(TCompactProtocol(TMemoryBuffer(data)))

class BufferWithPos(TMemoryBuffer):
    def getpos(self):return self._buffer.tell()

lib=ctypes.CDLL('libsnappy.so.1')
lib.snappy_uncompressed_length.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_size_t)]
lib.snappy_uncompressed_length.restype=ctypes.c_int
lib.snappy_uncompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t)]
lib.snappy_uncompress.restype=ctypes.c_int

def decompress(x,codec):
    if codec==0:return x
    if codec==1:
        cin=ctypes.create_string_buffer(x)
        n=ctypes.c_size_t()
        assert lib.snappy_uncompressed_length(cin, len(x), ctypes.byref(n))==0
        dst=ctypes.create_string_buffer(n.value)
        assert lib.snappy_uncompress(cin,len(x),dst,ctypes.byref(n))==0
        return dst.raw
    if codec==6: # ZSTD
        libzstd=ctypes.CDLL('libzstd.so.1')
        libzstd.ZSTD_getFrameContentSize.restype=ctypes.c_ulonglong
        libzstd.ZSTD_getFrameContentSize.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
        libzstd.ZSTD_decompress.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t]
        libzstd.ZSTD_decompress.restype=ctypes.c_size_t
        cin=ctypes.create_string_buffer(x)
        size=libzstd.ZSTD_getFrameContentSize(cin,len(x))
        if size>=2**64-2:raise ValueError('unknown ZSTD frame size')
        out=ctypes.create_string_buffer(size)
        n=libzstd.ZSTD_decompress(out,size,cin,len(x))
        if n!=size:raise ValueError(('zstd decoded size',n,size))
        return out.raw
    raise NotImplementedError('codec:'+str(codec))

def plain(data,ptype,n):
    if ptype==1:return np.frombuffer(data,dtype='<i4',count=n).copy()
    if ptype==2:return np.frombuffer(data,dtype='<i8',count=n).copy()
    if ptype==4:return np.frombuffer(data,dtype='<f4',count=n).copy()
    if ptype==5:return np.frombuffer(data,dtype='<f8',count=n).copy()
    if ptype==3: # INT96 timestamp nanoseconds parquet (julian day)
        b=np.frombuffer(data,dtype='u1',count=n*12).reshape(n,12)
        nano=np.frombuffer(np.ascontiguousarray(b[:,:8]).tobytes(),dtype='<i8')
        day=np.frombuffer(np.ascontiguousarray(b[:,8:]).tobytes(),dtype='<i4')
        return (day.astype('i8')-2440588)*86400000000000+nano
    if ptype==6:
        # byte array length-prefixed strings
        out=[]
        pos=0
        for i in range(n):
            ln=int.from_bytes(data[pos:pos+4],'little')
            out.append(data[pos+4:pos+4+ln])
            pos+=4+ln
        return np.asarray(out,dtype='object')
    raise NotImplementedError('ptype '+str(ptype))

def read_uleb(buf,pos):
    val=0;shift=0
    while True:
        x=buf[pos];pos+=1
        val|=(x&127)<<shift
        if not (x&128):return val,pos
        shift+=7

def decode_rle(buf,bit_width, nvals):
    out=np.empty(nvals,dtype='i8')
    pos=0;found=0
    while found<nvals:
        h,pos=read_uleb(buf,pos)
        if not h&1:
            run=h>>1
            nb=(bit_width+7)//8
            val=int.from_bytes(buf[pos:pos+nb],'little') if nb else 0
            pos+=nb
            count=min(run,nvals-found)
            out[found:found+count]=val
            found+=count
        else:
            groups=h>>1
            total=groups*8
            nbytes=groups*bit_width
            raw=buf[pos:pos+nbytes];pos+=nbytes
            cnt=min(total,nvals-found)
            for i in range(cnt):
                bit=i*bit_width
                byte=bit>>3
                shift=bit&7
                width=(bit_width+shift+7)>>3
                v=int.from_bytes(raw[byte:byte+width],'little')>>shift
                out[found+i]=v&((1<<bit_width)-1) if bit_width else 0
            found+=cnt
    return out

def decode_indices(buf,nvals):
    width=buf[0]
    return decode_rle(buf[1:],width,nvals)

def column_data(fp,colmeta, physical,optional):
    codec=colmeta[4]
    nvals=colmeta[5]
    start=colmeta.get(11,colmeta[9])
    fp.seek(start)
    values=[]; seen=0; dic=None
    while seen<nvals:
        hdrpos=fp.tell()
        # no fixed-size header, bounded amount of read input
        headbytes=fp.read(1024)
        p=BufferWithPos(headbytes)
        page=parse_struct(TCompactProtocol(p))
        consumed=p.getpos()
        fp.seek(hdrpos+consumed)
        csize=page[3]; usize=page[2]; kind=page[1]
        blob=fp.read(csize)
        if kind==2: # DICTIONARY_PAGE
            dh=page[7]
            dic=plain(decompress(blob,codec),physical,dh[1])
            continue
        if kind==0: # DATA_PAGE
            dph=page[5]
            page_n=dph[1]
            raw=decompress(blob,codec)
            offset=0
            if optional:
                # definition levels RLE with length prefix
                plen=struct.unpack_from('<I',raw,offset)[0];offset+=4
                levels=decode_rle(raw[offset:offset+plen],1,page_n)
                offset+=plen
                good=int(np.count_nonzero(levels))
            else:
                good=page_n
                levels=None
            encoding=dph[2]
            if encoding in (2,8):
                assert dic is not None
                v=dic[decode_indices(raw[offset:],good)]
            elif encoding==0:
                v=plain(raw[offset:],physical,good)
            else:raise NotImplementedError(('encode',encoding))
            if optional and good!=page_n:
                padded=np.full(page_n,np.nan,dtype='float64')
                padded[levels.astype(bool)]=v
                v=padded
            values.append(v)
            seen+=page_n
        elif kind==3: # DATA_PAGE_V2
            dh=page[8]
            page_n=dh[1]
            defs=dh.get(5,0);reps=dh.get(6,0)
            levels=decode_rle(blob[reps:reps+defs],1,page_n) if optional else None
            effective=page_n-int(dh.get(2,0))
            raw=(decompress(blob[reps+defs:],codec) if dh.get(7,True) else blob[reps+defs:])
            encoding=dh[4]
            if encoding in (2,8):
                assert dic is not None
                v=dic[decode_indices(raw,effective)]
            elif encoding==0:
                v=plain(raw,physical,effective)
            else:raise NotImplementedError(('encode',encoding))
            if optional and effective!=page_n:
                padded=np.full(page_n,np.nan,dtype='float64')
                padded[levels.astype(bool)]=v
                v=padded
            values.append(v)
            seen+=page_n
        else: raise NotImplementedError(('page type',kind))
    return np.concatenate(values)

def read_parquet_columns(path,names):
    md=read_meta(path)
    schema={e[4].decode('utf8'):e for e in md[2]}
    rows=[]
    with open(path,'rb') as fp:
        for rg in md[4]:
            colchunks=rg[1]
            cols={ch[3][3][0].decode('utf8'):ch[3] for ch in colchunks}
            out={}
            for name in names:
                if name not in cols: raise KeyError((name,list(cols)))
                out[name]=column_data(fp,cols[name],cols[name][1],schema[name].get(3,0)!=0)
            rows.append(out)
    return {n:np.concatenate([r[n] for r in rows]) for n in names}

if __name__=='__main__':
    import sys
    path=sys.argv[1]
    m=read_meta(path)
    print('rows',m[3], 'groups',len(m[4]))
    for elem in m[2]:
        print('schema',elem[4], 'physical',elem.get(1), 'opt',elem.get(3),'children',elem.get(5))
    for c in m[4][0][1]:
        print('col',c[3][3], 'codec',c[3][4], 'type',c[3][1], 'encodings',c[3][2], 'dpage',c[3].get(9), 'dictionary',c[3].get(11))
    readcols=read_parquet_columns(path,['lpep_pickup_datetime','PULocationID'])
    for n,v in readcols.items(): print('read',n,len(v),v.dtype,v[:15])
